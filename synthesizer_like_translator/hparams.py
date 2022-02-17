#from tensorflow.contrib.training import HParams
#from tensorboard.plugins.hparams import api as hp

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hyperparameter values."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numbers
import re

import six

#from tensorflow.contrib.training.python.training import hparam_pb2
from tensorflow.python.framework import ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation

# Define the regular expression for parsing a single clause of the input
# (delimited by commas).  A legal clause looks like:
#   <variable name>[<index>]? = <rhs>
# where <rhs> is either a single token or [] enclosed list of tokens.
# For example:  "var[1] = a" or "x = [1,2,3]"
PARAM_RE = re.compile(r"""
  (?P<name>[a-zA-Z][\w]*)      # variable name: "var" or "x"
  (\[\s*(?P<index>\d+)\s*\])?  # (optional) index: "1" or None
  \s*=\s*
  ((?P<val>[^,\[]*)            # single value: "a" or None
   |
   \[(?P<vals>[^\]]*)\])       # list of values: None or "1,2,3"
  ($|,\s*)""", re.VERBOSE)


def _parse_fail(name, var_type, value, values):
  """Helper function for raising a value error for bad assignment."""
  raise ValueError(
      'Could not parse hparam \'%s\' of type \'%s\' with value \'%s\' in %s' %
      (name, var_type.__name__, value, values))


def _reuse_fail(name, values):
  """Helper function for raising a value error for reuse of name."""
  raise ValueError('Multiple assignments to variable \'%s\' in %s' % (name,
                                                                      values))


def _process_scalar_value(name, parse_fn, var_type, m_dict, values,
                          results_dictionary):
  """Update results_dictionary with a scalar value.

  Used to update the results_dictionary to be returned by parse_values when
  encountering a clause with a scalar RHS (e.g.  "s=5" or "arr[0]=5".)

  Mutates results_dictionary.

  Args:
    name: Name of variable in assignment ("s" or "arr").
    parse_fn: Function for parsing the actual value.
    var_type: Type of named variable.
    m_dict: Dictionary constructed from regex parsing.
      m_dict['val']: RHS value (scalar)
      m_dict['index']: List index value (or None)
    values: Full expression being parsed
    results_dictionary: The dictionary being updated for return by the parsing
      function.

  Raises:
    ValueError: If the name has already been used.
  """
  try:
    parsed_value = parse_fn(m_dict['val'])
  except ValueError:
    _parse_fail(name, var_type, m_dict['val'], values)

  # If no index is provided
  if not m_dict['index']:
    if name in results_dictionary:
      _reuse_fail(name, values)
    results_dictionary[name] = parsed_value
  else:
    if name in results_dictionary:
      # The name has already been used as a scalar, then it
      # will be in this dictionary and map to a non-dictionary.
      if not isinstance(results_dictionary.get(name), dict):
        _reuse_fail(name, values)
    else:
      results_dictionary[name] = {}

    index = int(m_dict['index'])
    # Make sure the index position hasn't already been assigned a value.
    if index in results_dictionary[name]:
      _reuse_fail('{}[{}]'.format(name, index), values)
    results_dictionary[name][index] = parsed_value


def _process_list_value(name, parse_fn, var_type, m_dict, values,
                        results_dictionary):
  """Update results_dictionary from a list of values.

  Used to update results_dictionary to be returned by parse_values when
  encountering a clause with a list RHS (e.g.  "arr=[1,2,3]".)

  Mutates results_dictionary.

  Args:
    name: Name of variable in assignment ("arr").
    parse_fn: Function for parsing individual values.
    var_type: Type of named variable.
    m_dict: Dictionary constructed from regex parsing.
      m_dict['val']: RHS value (scalar)
    values: Full expression being parsed
    results_dictionary: The dictionary being updated for return by the parsing
      function.

  Raises:
    ValueError: If the name has an index or the values cannot be parsed.
  """
  if m_dict['index'] is not None:
    raise ValueError('Assignment of a list to a list index.')
  elements = filter(None, re.split('[ ,]', m_dict['vals']))
  # Make sure the name hasn't already been assigned a value
  if name in results_dictionary:
    raise _reuse_fail(name, values)
  try:
    results_dictionary[name] = [parse_fn(e) for e in elements]
  except ValueError:
    _parse_fail(name, var_type, m_dict['vals'], values)


def _cast_to_type_if_compatible(name, param_type, value):
  """Cast hparam to the provided type, if compatible.

  Args:
    name: Name of the hparam to be cast.
    param_type: The type of the hparam.
    value: The value to be cast, if compatible.

  Returns:
    The result of casting `value` to `param_type`.

  Raises:
    ValueError: If the type of `value` is not compatible with param_type.
      * If `param_type` is a string type, but `value` is not.
      * If `param_type` is a boolean, but `value` is not, or vice versa.
      * If `param_type` is an integer type, but `value` is not.
      * If `param_type` is a float type, but `value` is not a numeric type.
  """
  fail_msg = (
      "Could not cast hparam '%s' of type '%s' from value %r" %
      (name, param_type, value))

  # Some callers use None, for which we can't do any casting/checking. :(
  if issubclass(param_type, type(None)):
    return value

  # Avoid converting a non-string type to a string.
  if (issubclass(param_type, (six.string_types, six.binary_type)) and
      not isinstance(value, (six.string_types, six.binary_type))):
    raise ValueError(fail_msg)

  # Avoid converting a number or string type to a boolean or vice versa.
  if issubclass(param_type, bool) != isinstance(value, bool):
    raise ValueError(fail_msg)

  # Avoid converting float to an integer (the reverse is fine).
  if (issubclass(param_type, numbers.Integral) and
      not isinstance(value, numbers.Integral)):
    raise ValueError(fail_msg)

  # Avoid converting a non-numeric type to a numeric type.
  if (issubclass(param_type, numbers.Number) and
      not isinstance(value, numbers.Number)):
    raise ValueError(fail_msg)

  return param_type(value)


def parse_values(values, type_map):
  """Parses hyperparameter values from a string into a python map.

  `values` is a string containing comma-separated `name=value` pairs.
  For each pair, the value of the hyperparameter named `name` is set to
  `value`.

  If a hyperparameter name appears multiple times in `values`, a ValueError
  is raised (e.g. 'a=1,a=2', 'a[1]=1,a[1]=2').

  If a hyperparameter name in both an index assignment and scalar assignment,
  a ValueError is raised.  (e.g. 'a=[1,2,3],a[0] = 1').

  The `value` in `name=value` must follows the syntax according to the
  type of the parameter:

  *  Scalar integer: A Python-parsable integer point value.  E.g.: 1,
     100, -12.
  *  Scalar float: A Python-parsable floating point value.  E.g.: 1.0,
     -.54e89.
  *  Boolean: Either true or false.
  *  Scalar string: A non-empty sequence of characters, excluding comma,
     spaces, and square brackets.  E.g.: foo, bar_1.
  *  List: A comma separated list of scalar values of the parameter type
     enclosed in square brackets.  E.g.: [1,2,3], [1.0,1e-12], [high,low].

  When index assignment is used, the corresponding type_map key should be the
  list name.  E.g. for "arr[1]=0" the type_map must have the key "arr" (not
  "arr[1]").

  Args:
    values: String.  Comma separated list of `name=value` pairs where
      'value' must follow the syntax described above.
    type_map: A dictionary mapping hyperparameter names to types.  Note every
      parameter name in values must be a key in type_map.  The values must
      conform to the types indicated, where a value V is said to conform to a
      type T if either V has type T, or V is a list of elements of type T.
      Hence, for a multidimensional parameter 'x' taking float values,
      'x=[0.1,0.2]' will parse successfully if type_map['x'] = float.

  Returns:
    A python map mapping each name to either:
    * A scalar value.
    * A list of scalar values.
    * A dictionary mapping index numbers to scalar values.
    (e.g. "x=5,L=[1,2],arr[1]=3" results in {'x':5,'L':[1,2],'arr':{1:3}}")

  Raises:
    ValueError: If there is a problem with input.
    * If `values` cannot be parsed.
    * If a list is assigned to a list index (e.g. 'a[1] = [1,2,3]').
    * If the same rvalue is assigned two different values (e.g. 'a=1,a=2',
      'a[1]=1,a[1]=2', or 'a=1,a=[1]')
  """
  results_dictionary = {}
  pos = 0
  while pos < len(values):
    m = PARAM_RE.match(values, pos)
    if not m:
      raise ValueError('Malformed hyperparameter value: %s' % values[pos:])
    # Check that there is a comma between parameters and move past it.
    pos = m.end()
    # Parse the values.
    m_dict = m.groupdict()
    name = m_dict['name']
    if name not in type_map:
      raise ValueError('Unknown hyperparameter type for %s' % name)
    type_ = type_map[name]

    # Set up correct parsing function (depending on whether type_ is a bool)
    if type_ == bool:

      def parse_bool(value):
        if value in ['true', 'True']:
          return True
        elif value in ['false', 'False']:
          return False
        else:
          try:
            return bool(int(value))
          except ValueError:
            _parse_fail(name, type_, value, values)

      parse = parse_bool
    else:
      parse = type_

    # If a singe value is provided
    if m_dict['val'] is not None:
      _process_scalar_value(name, parse, type_, m_dict, values,
                            results_dictionary)

    # If the assigned value is a list:
    elif m_dict['vals'] is not None:
      _process_list_value(name, parse, type_, m_dict, values,
                          results_dictionary)

    else:  # Not assigned a list or value
      _parse_fail(name, type_, '', values)

  return results_dictionary


class HParams(object):
  """Class to hold a set of hyperparameters as name-value pairs.

  A `HParams` object holds hyperparameters used to build and train a model,
  such as the number of hidden units in a neural net layer or the learning rate
  to use when training.

  You first create a `HParams` object by specifying the names and values of the
  hyperparameters.

  To make them easily accessible the parameter names are added as direct
  attributes of the class.  A typical usage is as follows:

  ```python
  # Create a HParams object specifying names and values of the model
  # hyperparameters:
  hparams = HParams(learning_rate=0.1, num_hidden_units=100)

  # The hyperparameter are available as attributes of the HParams object:
  hparams.learning_rate ==> 0.1
  hparams.num_hidden_units ==> 100
  ```

  Hyperparameters have type, which is inferred from the type of their value
  passed at construction type.   The currently supported types are: integer,
  float, string, and list of integer, float, or string.

  You can override hyperparameter values by calling the
  [`parse()`](#HParams.parse) method, passing a string of comma separated
  `name=value` pairs.  This is intended to make it possible to override
  any hyperparameter values from a single command-line flag to which
  the user passes 'hyper-param=value' pairs.  It avoids having to define
  one flag for each hyperparameter.

  The syntax expected for each value depends on the type of the parameter.
  See `parse()` for a description of the syntax.

  Example:

  ```python
  # Define a command line flag to pass name=value pairs.
  # For example using argparse:
  import argparse
  parser = argparse.ArgumentParser(description='Train my model.')
  parser.add_argument('--hparams', type=str,
                      help='Comma separated list of "name=value" pairs.')
  args = parser.parse_args()
  ...
  def my_program():
    # Create a HParams object specifying the names and values of the
    # model hyperparameters:
    hparams = tf.HParams(learning_rate=0.1, num_hidden_units=100,
                         activations=['relu', 'tanh'])

    # Override hyperparameters values by parsing the command line
    hparams.parse(args.hparams)

    # If the user passed `--hparams=learning_rate=0.3` on the command line
    # then 'hparams' has the following attributes:
    hparams.learning_rate ==> 0.3
    hparams.num_hidden_units ==> 100
    hparams.activations ==> ['relu', 'tanh']

    # If the hyperparameters are in json format use parse_json:
    hparams.parse_json('{"learning_rate": 0.3, "activations": "relu"}')
  ```
  """

  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self, hparam_def=None, model_structure=None, **kwargs):
    """Create an instance of `HParams` from keyword arguments.

    The keyword arguments specify name-values pairs for the hyperparameters.
    The parameter types are inferred from the type of the values passed.

    The parameter names are added as attributes of `HParams` object, so they
    can be accessed directly with the dot notation `hparams._name_`.

    Example:

    ```python
    # Define 3 hyperparameters: 'learning_rate' is a float parameter,
    # 'num_hidden_units' an integer parameter, and 'activation' a string
    # parameter.
    hparams = tf.HParams(
        learning_rate=0.1, num_hidden_units=100, activation='relu')

    hparams.activation ==> 'relu'
    ```

    Note that a few names are reserved and cannot be used as hyperparameter
    names.  If you use one of the reserved name the constructor raises a
    `ValueError`.

    Args:
      hparam_def: Serialized hyperparameters, encoded as a hparam_pb2.HParamDef
        protocol buffer. If provided, this object is initialized by
        deserializing hparam_def.  Otherwise **kwargs is used.
      model_structure: An instance of ModelStructure, defining the feature
        crosses to be used in the Trial.
      **kwargs: Key-value pairs where the key is the hyperparameter name and
        the value is the value for the parameter.

    Raises:
      ValueError: If both `hparam_def` and initialization values are provided,
        or if one of the arguments is invalid.

    """
    # Register the hyperparameters and their type in _hparam_types.
    # This simplifies the implementation of parse().
    # _hparam_types maps the parameter name to a tuple (type, bool).
    # The type value is the type of the parameter for scalar hyperparameters,
    # or the type of the list elements for multidimensional hyperparameters.
    # The bool value is True if the value is a list, False otherwise.
    self._hparam_types = {}
    self._model_structure = model_structure
    if hparam_def:
      self._init_from_proto(hparam_def)
      if kwargs:
        raise ValueError('hparam_def and initialization values are '
                         'mutually exclusive')
    else:
      for name, value in six.iteritems(kwargs):
        self.add_hparam(name, value)

  def _init_from_proto(self, hparam_def):
    """Creates a new HParams from `HParamDef` protocol buffer.

    Args:
      hparam_def: `HParamDef` protocol buffer.
    """
    #assert isinstance(hparam_def, hparam_pb2.HParamDef)
    for name, value in hparam_def.hparam.items():
      kind = value.WhichOneof('kind')
      if kind.endswith('_value'):
        # Single value.
        if kind.startswith('int64'):
          # Setting attribute value to be 'int' to ensure the type is compatible
          # with both Python2 and Python3.
          self.add_hparam(name, int(getattr(value, kind)))
        elif kind.startswith('bytes'):
          # Setting attribute value to be 'str' to ensure the type is compatible
          # with both Python2 and Python3. UTF-8 encoding is assumed.
          self.add_hparam(name, compat.as_str(getattr(value, kind)))
        else:
          self.add_hparam(name, getattr(value, kind))
      else:
        # List of values.
        if kind.startswith('int64'):
          # Setting attribute value to be 'int' to ensure the type is compatible
          # with both Python2 and Python3.
          self.add_hparam(name, [int(v) for v in getattr(value, kind).value])
        elif kind.startswith('bytes'):
          # Setting attribute value to be 'str' to ensure the type is compatible
          # with both Python2 and Python3. UTF-8 encoding is assumed.
          self.add_hparam(
              name, [compat.as_str(v) for v in getattr(value, kind).value])
        else:
          self.add_hparam(name, [v for v in getattr(value, kind).value])

  def add_hparam(self, name, value):
    """Adds {name, value} pair to hyperparameters.

    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter. Can be one of the following types:
        int, float, string, int list, float list, or string list.

    Raises:
      ValueError: if one of the arguments is invalid.
    """
    # Keys in kwargs are unique, but 'name' could the name of a pre-existing
    # attribute of this object.  In that case we refuse to use it as a
    # hyperparameter name.
    if getattr(self, name, None) is not None:
      raise ValueError('Hyperparameter name is reserved: %s' % name)
    if isinstance(value, (list, tuple)):
      if not value:
        raise ValueError(
            'Multi-valued hyperparameters cannot be empty: %s' % name)
      self._hparam_types[name] = (type(value[0]), True)
    else:
      self._hparam_types[name] = (type(value), False)
    setattr(self, name, value)

  def set_hparam(self, name, value):
    """Set the value of an existing hyperparameter.

    This function verifies that the type of the value matches the type of the
    existing hyperparameter.

    Args:
      name: Name of the hyperparameter.
      value: New value of the hyperparameter.

    Raises:
      ValueError: If there is a type mismatch.
    """
    param_type, is_list = self._hparam_types[name]
    if isinstance(value, list):
      if not is_list:
        raise ValueError(
            'Must not pass a list for single-valued parameter: %s' % name)
      setattr(self, name, [
          _cast_to_type_if_compatible(name, param_type, v) for v in value])
    else:
      if is_list:
        raise ValueError(
            'Must pass a list for multi-valued parameter: %s.' % name)
      setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))

  def parse(self, values):
    """Override hyperparameter values, parsing new values from a string.

    See parse_values for more detail on the allowed format for values.

    Args:
      values: String.  Comma separated list of `name=value` pairs where
        'value' must follow the syntax described above.

    Returns:
      The `HParams` instance.

    Raises:
      ValueError: If `values` cannot be parsed.
    """
    type_map = dict()
    for name, t in self._hparam_types.items():
      param_type, _ = t
      type_map[name] = param_type

    values_map = parse_values(values, type_map)
    return self.override_from_dict(values_map)

  def override_from_dict(self, values_dict):
    """Override hyperparameter values, parsing new values from a dictionary.

    Args:
      values_dict: Dictionary of name:value pairs.

    Returns:
      The `HParams` instance.

    Raises:
      ValueError: If `values_dict` cannot be parsed.
    """
    for name, value in values_dict.items():
      self.set_hparam(name, value)
    return self

  @deprecation.deprecated(None, 'Use `override_from_dict`.')
  def set_from_map(self, values_map):
    """DEPRECATED. Use override_from_dict."""
    return self.override_from_dict(values_dict=values_map)

  def set_model_structure(self, model_structure):
    self._model_structure = model_structure

  def get_model_structure(self):
    return self._model_structure

  def to_json(self, indent=None, separators=None, sort_keys=False):
    """Serializes the hyperparameters into JSON.

    Args:
      indent: If a non-negative integer, JSON array elements and object members
        will be pretty-printed with that indent level. An indent level of 0, or
        negative, will only insert newlines. `None` (the default) selects the
        most compact representation.
      separators: Optional `(item_separator, key_separator)` tuple. Default is
        `(', ', ': ')`.
      sort_keys: If `True`, the output dictionaries will be sorted by key.

    Returns:
      A JSON string.
    """
    return json.dumps(
        self.values(),
        indent=indent,
        separators=separators,
        sort_keys=sort_keys)

  def parse_json(self, values_json):
    """Override hyperparameter values, parsing new values from a json object.

    Args:
      values_json: String containing a json object of name:value pairs.

    Returns:
      The `HParams` instance.

    Raises:
      ValueError: If `values_json` cannot be parsed.
    """
    values_map = json.loads(values_json)
    return self.override_from_dict(values_map)

  def values(self):
    """Return the hyperparameter values as a Python dictionary.

    Returns:
      A dictionary with hyperparameter names as keys.  The values are the
      hyperparameter values.
    """
    return {n: getattr(self, n) for n in self._hparam_types.keys()}

  def get(self, key, default=None):
    """Returns the value of `key` if it exists, else `default`."""
    if key in self._hparam_types:
      # Ensure that default is compatible with the parameter type.
      if default is not None:
        param_type, is_param_list = self._hparam_types[key]
        type_str = 'list<%s>' % param_type if is_param_list else str(param_type)
        fail_msg = ("Hparam '%s' of type '%s' is incompatible with "
                    'default=%s' % (key, type_str, default))

        is_default_list = isinstance(default, list)
        if is_param_list != is_default_list:
          raise ValueError(fail_msg)

        try:
          if is_default_list:
            for value in default:
              _cast_to_type_if_compatible(key, param_type, value)
          else:
            _cast_to_type_if_compatible(key, param_type, default)
        except ValueError as e:
          raise ValueError('%s. %s' % (fail_msg, e))

      return getattr(self, key)

    return default

  def __contains__(self, key):
    return key in self._hparam_types

  def __str__(self):
    return str(sorted(self.values().items()))

  def __repr__(self):
    return '%s(%s)' % (type(self).__name__, self.__str__())

  @staticmethod
  def _get_kind_name(param_type, is_list):
    """Returns the field name given parameter type and is_list.

    Args:
      param_type: Data type of the hparam.
      is_list: Whether this is a list.

    Returns:
      A string representation of the field name.

    Raises:
      ValueError: If parameter type is not recognized.
    """
    if issubclass(param_type, bool):
      # This check must happen before issubclass(param_type, six.integer_types),
      # since Python considers bool to be a subclass of int.
      typename = 'bool'
    elif issubclass(param_type, six.integer_types):
      # Setting 'int' and 'long' types to be 'int64' to ensure the type is
      # compatible with both Python2 and Python3.
      typename = 'int64'
    elif issubclass(param_type, (six.string_types, six.binary_type)):
      # Setting 'string' and 'bytes' types to be 'bytes' to ensure the type is
      # compatible with both Python2 and Python3.
      typename = 'bytes'
    elif issubclass(param_type, float):
      typename = 'float'
    else:
      raise ValueError('Unsupported parameter type: %s' % str(param_type))

    suffix = 'list' if is_list else 'value'
    return '_'.join([typename, suffix])

#   def to_proto(self, export_scope=None):  # pylint: disable=unused-argument
#     """Converts a `HParams` object to a `HParamDef` protocol buffer.

#     Args:
#       export_scope: Optional `string`. Name scope to remove.

#     Returns:
#       A `HParamDef` protocol buffer.
#     """
#     hparam_proto = hparam_pb2.HParamDef()
#     for name in self._hparam_types:
#       # Parse the values.
#       param_type, is_list = self._hparam_types.get(name, (None, None))
#       kind = HParams._get_kind_name(param_type, is_list)

#       if is_list:
#         if kind.startswith('bytes'):
#           v_list = [compat.as_bytes(v) for v in getattr(self, name)]
#         else:
#           v_list = [v for v in getattr(self, name)]
#         getattr(hparam_proto.hparam[name], kind).value.extend(v_list)
#       else:
#         v = getattr(self, name)
#         if kind.startswith('bytes'):
#           v = compat.as_bytes(getattr(self, name))
#         setattr(hparam_proto.hparam[name], kind, v)

#     return hparam_proto

#   @staticmethod
#   def from_proto(hparam_def, import_scope=None):  # pylint: disable=unused-argument
#     return HParams(hparam_def=hparam_def)


# ops.register_proto_function(
#     'hparams',
#     proto_type=hparam_pb2.HParamDef,
#     to_proto=HParams.to_proto,
#     from_proto=HParams.from_proto)


# Default hyperparameters
hparams = HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners".
    cleaners="english_cleaners",
    
    # If you only have 1 GPU or want to use only one GPU, please set num_gpus=0 and specify the 
    # GPU idx on run. example:
    # expample 1 GPU of index 2 (train on "/gpu2" only): CUDA_VISIBLE_DEVICES=2 python train.py 
    # --model="Tacotron" --hparams="tacotron_gpu_start_idx=2"
    # If you want to train on multiple GPUs, simply specify the number of GPUs available, 
    # and the idx of the first GPU to use. example:
    # example 4 GPUs starting from index 0 (train on "/gpu0"->"/gpu3"): python train.py 
    # --model="Tacotron" --hparams="tacotron_num_gpus=4, tacotron_gpu_start_idx=0"
    # The hparams arguments can be directly modified on this hparams.py file instead of being 
    # specified on run if preferred!
    
    # If one wants to train both Tacotron and WaveNet in parallel (provided WaveNet will be 
    # trained on True mel spectrograms), one needs to specify different GPU idxes.
    # example Tacotron+WaveNet on a machine with 4 or plus GPUs. Two GPUs for each model: 
    # CUDA_VISIBLE_DEVICES=0,1 python train.py --model="Tacotron" 
	# --hparams="tacotron_gpu_start_idx=0, tacotron_num_gpus=2"
    # Cuda_VISIBLE_DEVICES=2,3 python train.py --model="WaveNet" 
	# --hparams="wavenet_gpu_start_idx=2; wavenet_num_gpus=2"
    
    # IMPORTANT NOTE: If using N GPUs, please multiply the tacotron_batch_size by N below in the 
    # hparams! (tacotron_batch_size = 32 * N)
    # Never use lower batch size than 32 on a single GPU!
    # Same applies for Wavenet: wavenet_batch_size = 8 * N (wavenet_batch_size can be smaller than
    #  8 if GPU is having OOM, minimum 2)
    # Please also apply the synthesis batch size modification likewise. (if N GPUs are used for 
    # synthesis, minimal batch size must be N, minimum of 1 sample per GPU)
    # We did not add an automatic multi-GPU batch size computation to avoid confusion in the 
    # user"s mind and to provide more control to the user for
    # resources related decisions.
    
    # Acknowledgement:
    #	Many thanks to @MlWoo for his awesome work on multi-GPU Tacotron which showed to work a 
	# little faster than the original
    #	pipeline for a single GPU as well. Great work!
    
    # Hardware setup: Default supposes user has only one GPU: "/gpu:0" (Tacotron only for now! 
    # WaveNet does not support multi GPU yet, WIP)
    # Synthesis also uses the following hardware parameters for multi-GPU parallel synthesis.
    tacotron_gpu_start_idx=0,  # idx of the first GPU to be used for Tacotron training.
    tacotron_num_gpus=2,  # Determines the number of gpus in use for Tacotron training.
    split_on_cpu=True,
    # Determines whether to split data on CPU or on first GPU. This is automatically True when 
	# more than 1 GPU is used.
    ###########################################################################################################################################
    
    # Audio
    # Audio parameters are the most important parameters to tune when using this work on your 
    # personal data. Below are the beginner steps to adapt
    # this work to your personal data:
    #	1- Determine my data sample rate: First you need to determine your audio sample_rate (how 
	# many samples are in a second of audio). This can be done using sox: "sox --i <filename>"
    #		(For this small tuto, I will consider 24kHz (24000 Hz), and defaults are 22050Hz, 
	# so there are plenty of examples to refer to)
    #	2- set sample_rate parameter to your data correct sample rate
    #	3- Fix win_size and and hop_size accordingly: (Supposing you will follow our advice: 50ms 
	# window_size, and 12.5ms frame_shift(hop_size))
    #		a- win_size = 0.05 * sample_rate. In the tuto example, 0.05 * 24000 = 1200
    #		b- hop_size = 0.25 * win_size. Also equal to 0.0125 * sample_rate. In the tuto 
	# example, 0.25 * 1200 = 0.0125 * 24000 = 300 (Can set frame_shift_ms=12.5 instead)
    #	4- Fix n_fft, num_freq and upsample_scales parameters accordingly.
    #		a- n_fft can be either equal to win_size or the first power of 2 that comes after 
	# win_size. I usually recommend using the latter
    #			to be more consistent with signal processing friends. No big difference to be seen
	#  however. For the tuto example: n_fft = 2048 = 2**11
    #		b- num_freq = (n_fft / 2) + 1. For the tuto example: num_freq = 2048 / 2 + 1 = 1024 + 
	# 1 = 1025.
    #		c- For WaveNet, upsample_scales products must be equal to hop_size. For the tuto 
	# example: upsample_scales=[15, 20] where 15 * 20 = 300
    #			it is also possible to use upsample_scales=[3, 4, 5, 5] instead. One must only 
	# keep in mind that upsample_kernel_size[0] = 2*upsample_scales[0]
    #			so the training segments should be long enough (2.8~3x upsample_scales[0] * 
	# hop_size or longer) so that the first kernel size can see the middle 
    #			of the samples efficiently. The length of WaveNet training segments is under the 
	# parameter "max_time_steps".
    #	5- Finally comes the silence trimming. This very much data dependent, so I suggest trying 
	# preprocessing (or part of it, ctrl-C to stop), then use the
    #		.ipynb provided in the repo to listen to some inverted mel/linear spectrograms. That 
	# will first give you some idea about your above parameters, and
    #		it will also give you an idea about trimming. If silences persist, try reducing 
	# trim_top_db slowly. If samples are trimmed mid words, try increasing it.
    #	6- If audio quality is too metallic or fragmented (or if linear spectrogram plots are 
	# showing black silent regions on top), then restart from step 2.
    num_mels=256,  # Number of mel-spectrogram channels and local conditioning dimensionality    
    use_full_ppg=False, # If use full ppg, if yes, num_ppgs should be 5816
    num_ppgs=256,  # Number of PPG channels
    #  network
    rescale=False,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value
    # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    # train samples of lengths between 3sec and 14sec are more than enough to make a model capable
    # of good parallelization.
    clip_mels_length=True,
    # For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, 
	# also consider clipping your samples to smaller chunks)
    max_mel_frames=900,
    # Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3
	#  and still getting OOM errors.
    
    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,
    # Only used to set as True if using WaveNet, no difference in performance is observed in 
    # either cases.
    silence_threshold=2,  # silence threshold used for sound trimming for wavenet preprocessing
    
    # Mel spectrogram  
    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    
    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
    
    # M-AILABS (and other datasets) trim params (these parameters are usually correct for any 
	# data, but definitely must be tuned for specific speakers)
    trim_fft_size=512,
    trim_hop_size=128,
    trim_top_db=23,
    
    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
    # be too big to avoid gradient explosion, 
    # not too small for fast convergence)
    normalize_for_wavenet=True,
    # whether to rescale to [0, 1] for wavenet. (better audio quality)
    clip_for_wavenet=True,
    # whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
    
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
	# levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.
    
    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
	# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.
    
    # Griffin Lim
    power=1.5,
    # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    griffin_lim_iters=60,
    # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
    ###########################################################################################################################################
    
    # Tacotron
    outputs_per_step=1, # Was 1
    # number of frames to generate at each decoding step (increase to speed up computation and 
    # allows for higher batch size, decreases G&L audio quality)
    stop_at_any=True,
    # Determines whether the decoder should stop when predicting <stop> to any frame or to all of 
    # them (True works pretty well)
    
    embedding_dim=512,  # dimension of embedding space (these are NOT the speaker embeddings)
    
    # Encoder parameters
    enc_prenet_layers=[0, 0], # number of layers and number of units of encoder prenet
    enc_conv_num_layers=3,  # number of encoder convolutional layers
    enc_conv_kernel_size=(5,),  # size of encoder convolution filters for each layer
    enc_conv_channels=512,  # number of encoder convolutions filters for each layer
    encoder_lstm_units=256,  # number of lstm units for each direction (forward and backward)
    is_encoder_lstm_pyramid=True, # if use pyramid lstm to downsample the number of outputs of encoder
    is_encoder_lstm_2layers=False, # if use two layers of pyramid lstm

    # Attention mechanism
    smoothing=False,  # Whether to smooth the attention normalization function
    attention_dim=512,  # dimension of attention space
    attention_filters=32,  # number of attention convolution filters
    attention_kernel=(31,),  # kernel size of attention convolution
    cumulative_weights=True,
    # Whether to cumulate (sum) all previous attention weights or simply feed previous weights (
    # Recommended: True)
    
    # Decoder
    prenet_layers=[512, 512],  # number of layers and number of units of prenet
    decoder_layers=2,  # number of decoder lstm layers
    decoder_lstm_units=2048,  # number of decoder lstm units on each layer
    max_iters=2000,
    # Max decoder steps during inference (Just for safety from infinite loop cases)
    
    # Residual postnet
    postnet_num_layers=5,  # number of postnet convolutional layers
    postnet_kernel_size=(5,),  # size of postnet convolution filters for each layer
    postnet_channels=1024,  # number of postnet convolution filters for each layer
    
    # CBHG mel->linear postnet
    cbhg_kernels=8,
    # All kernel sizes from 1 to cbhg_kernels will be used in the convolution bank of CBHG to act
    #  as "K-grams"
    cbhg_conv_channels=128,  # Channels of the convolution bank
    cbhg_pool_size=2,  # pooling size of the CBHG
    cbhg_projection=256,
    # projection channels of the CBHG (1st projection, 2nd is automatically set to num_mels)
    cbhg_projection_kernel_size=3,  # kernel_size of the CBHG projections
    cbhg_highwaynet_layers=4,  # Number of HighwayNet layers
    cbhg_highway_units=512,  # Number of units used in HighwayNet fully connected layers
    cbhg_rnn_units=512,
    # Number of GRU units used in bidirectional RNN of CBHG block. CBHG output is 2x rnn_units in 
    # shape
    
    # Loss params
    mask_encoder=True,
    # whether to mask encoder padding while computing attention. Set to True for better prosody 
    # but slower convergence.
    mask_decoder=False,
    # Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not
    #  be weighted, else recommended pos_weight = 20)
    cross_entropy_pos_weight=20,
    # Use class weights to reduce the stop token classes imbalance (by adding more penalty on 
    # False Negatives (FN)) (1 = disabled)
    predict_linear=False,
    # Whether to add a post-processing network to the Tacotron to predict linear spectrograms (
	# True mode Not tested!!)
    ###########################################################################################################################################

    # Tacotron Training
    # Reproduction seeds
    tacotron_random_seed=5339,
    # Determines initial graph and operations (i.e: model) random state for reproducibility
    tacotron_data_random_state=1234,  # random state for train test split repeatability
    
    # performance parameters
    tacotron_swap_with_cpu=False,
    # Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause 
    # major slowdowns! Only use when critical!)
    
    # train/test split ratios, mini-batches sizes
    tacotron_batch_size=32,  # number of training samples on each training steps (was 32)
    # Tacotron Batch synthesis supports ~16x the training batch size (no gradients during 
    # testing). 
    # Training Tacotron with unmasked paddings makes it aware of them, which makes synthesis times
    #  different from training. We thus recommend masking the encoder.
    tacotron_synthesis_batch_size=128,
    # DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN"T TRAIN TACOTRON WITH "mask_encoder=True"!!
    tacotron_test_size=0.05,
    # % of data to keep as test data, if None, tacotron_test_batches must be not None. (5% is 
	# enough to have a good idea about overfit)
    tacotron_test_batches=None,  # number of test batches.
    
    # Learning rate schedule
    tacotron_decay_learning_rate=True,
    # boolean, determines if the learning rate will follow an exponential decay
    tacotron_start_decay=50000,  # Step at which learning decay starts
    tacotron_decay_steps=50000,  # Determines the learning rate decay slope (UNDER TEST)
    tacotron_decay_rate=0.5,  # learning rate decay rate (UNDER TEST)
    tacotron_initial_learning_rate=1e-3,  # starting learning rate
    tacotron_final_learning_rate=1e-5,  # minimal learning rate
    
    # Optimization parameters
    tacotron_adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    tacotron_adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    tacotron_adam_epsilon=1e-6,  # AdamOptimizer Epsilon parameter
    
    # Regularization parameters
    tacotron_reg_weight=1e-7,  # regularization weight (for L2 regularization)
    tacotron_scale_regularization=False,
    # Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is
    #  high and biasing the model)
    tacotron_zoneout_rate=0.1,  # zoneout rate for all LSTM cells in the network
    tacotron_dropout_rate=0.5,  # dropout rate for all convolutional layers + prenet
    tacotron_clip_gradients=True,  # whether to clip gradients
    
    # Evaluation parameters
    natural_eval=False,
    # Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same
	#  teacher-forcing ratio as in training (just for overfit)
    
    # Decoder RNN learning can take be done in one of two ways:
    #	Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode="constant"
    #	Curriculum Learning Scheme: From Teacher-Forcing to sampling from previous outputs is 
    # function of global step. (teacher forcing ratio decay) mode="scheduled"
    # The second approach is inspired by:
    # Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
    # Can be found under: https://arxiv.org/pdf/1506.03099.pdf
    tacotron_teacher_forcing_mode="constant",
    # Can be ("constant" or "scheduled"). "scheduled" mode applies a cosine teacher forcing ratio 
    # decay. (Preference: scheduled)
    tacotron_teacher_forcing_ratio=1.,
    # Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder 
	# inputs, Only relevant if mode="constant"
    tacotron_teacher_forcing_init_ratio=1.,
    # initial teacher forcing ratio. Relevant if mode="scheduled"
    tacotron_teacher_forcing_final_ratio=0.,
    # final teacher forcing ratio. Relevant if mode="scheduled"
    tacotron_teacher_forcing_start_decay=10000,
    # starting point of teacher forcing ratio decay. Relevant if mode="scheduled"
    tacotron_teacher_forcing_decay_steps=280000,
    # Determines the teacher forcing ratio decay slope. Relevant if mode="scheduled"
    tacotron_teacher_forcing_decay_alpha=0.,
    # teacher forcing ratio decay rate. Relevant if mode="scheduled"
    ###########################################################################################################################################
 
    # Tacotron-2 integration parameters
    train_with_GTA=False,
    # Whether to use GTA mels to train WaveNet instead of ground truth mels.
    ###########################################################################################################################################
    
    # Eval sentences (if no eval text file was specified during synthesis, these sentences are 
	# used for eval)
    sentences=[
        # From July 8, 2017 New York Times:
        "Scientists at the CERN laboratory say they have discovered a new particle.",
        "There\"s a way to measure the acute emotional intelligence that has never gone out of "
		"style.",
        "President Trump met with other leaders at the Group of 20 conference.",
        "The Senate\"s bill to repeal and replace the Affordable Care Act is now imperiled.",
        # From Google"s Tacotron example page:
        "Generative adversarial network or variational auto-encoder.",
        "Basilar membrane and otolaryngology are not auto-correlations.",
        "He has read the whole thing.",
        "He reads books.",
        "He thought it was time to present the present.",
        "Thisss isrealy awhsome.",
        "Punctuation sensitivity, is working.",
        "Punctuation sensitivity is working.",
        "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
        "She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
        "Tajima Airport serves Toyooka.",
        # From The web (random long utterance)
        "Sequence to sequence models have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization.\
        This project covers a sequence to sequence model trained to predict a speech representation from an input sequence of characters. We show that\
        the adopted architecture is able to perform this task with wild success.",
        "Thank you so much for your support!",
    ],
    
    
    ### SV2TTS ###
    speaker_embedding_size=256,
    silence_min_duration_split=0.4, # Duration in seconds of a silence for an utterance to be split
    utterance_min_duration=1.6,     # Duration in seconds below which utterances are discarded
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)