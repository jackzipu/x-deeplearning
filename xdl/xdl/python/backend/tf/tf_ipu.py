import re
import tensorflow as tf
from tensorflow.python import ipu

from tensorflow.core.framework import attr_value_pb2

def tf_ipu_graph(cur_graph):
    _init_ipu_config()
    _graph_def = cur_graph.as_graph_def()
    _do_ipu_placement(_graph_def)
    _imported_graph = tf.Graph()
    with _imported_graph.as_default():
        tf.import_graph_def(_graph_def, name="")
    return _imported_graph

_ipu_initialized=False
def _init_ipu_config():
    global _ipu_initialized
    if _ipu_initialized:
        return
    # Create a default configuration
    ipu_configuration = ipu.utils.create_ipu_config()
    # Select an IPU automatically
    ipu_configuration = ipu.utils.auto_select_ipus(opts=ipu_configuration, num_ipus=1)
    # Apply the configuration
    ipu.utils.configure_ipu_system(config=ipu_configuration)
    _ipu_initialized = True

def _do_ipu_placement(graph_def):
    for i, node in enumerate(graph_def.node):
      if _should_do_placement(node):
        _add_ipu_scope(node)

def _add_ipu_scope(node):
    node.device = '/device:IPU:0'
    node.attr['_XlaCompile'].CopyFrom(attr_value_pb2.AttrValue(b=True))
    node.attr['_XlaScope'].CopyFrom(
        attr_value_pb2.AttrValue(s='jit_scope_ipu_0'.encode()))
    node.attr['_XlaSeparateCompiledGradients'].CopyFrom(
        attr_value_pb2.AttrValue(b=False))

#node_blacklist = [".*Initializer.*", "^dense.{0,2}/kernel$", "^dense.{0,2}/kernel/Assign$", "^dense.{0,2}/kernel/read$",]
node_blacklist = []
def _should_do_placement(node):
    global node_blacklist
    for pattern in node_blacklist:
      if re.search(pattern, node.name):
        return False

    if hasattr(node, 'device') and node.op != 'Placeholder':
      return True

    return False