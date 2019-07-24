
"""
A MultistepAdamWeightDecayOptimizer can use larger batch_size in BERT 
which updates var after n steps 

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.training import optimizer
from tensorflow.python.framework import ops


class MultistepAdamWeightDecayOptimizer(optimizer.Optimizer):
  """A Multistep Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate=0.01,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               n=1, ##n steps per update
               exclude_from_weight_decay=None,
               name="MultistepAdamWeightDecayOptimizer"):
    """Constructs a MultistepAdamWeightDecayOptimizer."""
    super(MultistepAdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self._n = n  # Call Adam optimizer every n batches with accumulated grads
    self.exclude_from_weight_decay = exclude_from_weight_decay

    self._n_t = None  # n as tensor

  def _prepare(self):
    super(MultistepAdamWeightDecayOptimizer, self)._prepare()
    self._n_t = tf.convert_to_tensor(self._n, name="n")

  def _create_slots(self, var_list):
    """Create slot variables for MultistepAdamWeightDecayOptimizer with accumulated gradients.

    Like super class method, but additionally creates slots for the gradient
    accumulator `grad_acc` and the counter variable.
    """
    super(MultistepAdamWeightDecayOptimizer, self)._create_slots(var_list)
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=0 if self._n == 1 else 1,
                                   name="iter",
                                   colocate_with=first_var)
    for v in var_list:
      self._zeros_slot(v, "grad_acc", self._name)

  def _get_iter_variable(self):
    if tf.contrib.eager.in_eager_mode():
      graph = None
    else:
      graph = tf.get_default_graph()
    return self._get_non_slot_variable("iter", graph=graph)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    update_ops = []

    var_list = [v for g, v in grads_and_vars if g is not None]

    with ops.init_scope():
      self._create_slots(var_list)
    self._prepare()

    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue
      grad_acc = self.get_slot(param, "grad_acc")
      param_name = self._get_variable_name(param.name)
      m = tf.get_variable(name=param_name + "/adam_m",shape=param.shape.as_list(),
          dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer())
      v = tf.get_variable(name=param_name + "/adam_v",shape=param.shape.as_list(),
          dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer())
      
      ##apply adam for v
      def _apply_adam(grad_acc, grad, param, m, v):
        total_grad = (grad_acc + grad) / tf.cast(self._n_t, grad.dtype)
        # Standard Adam update.
        next_m = (
            tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, total_grad))
        next_v = (
            tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                      tf.square(total_grad)))
        update = next_m / (tf.sqrt(next_v) + self.epsilon)
        if self._do_use_weight_decay(param_name):
          update += self.weight_decay_rate * param
        update_with_lr = self.learning_rate * update
        next_param = param - update_with_lr
        adam_op = tf.group(param.assign(next_param), m.assign(next_m),
                        v.assign(next_v))
        with tf.control_dependencies([adam_op]):
          grad_acc_to_zero_op = grad_acc.assign(tf.zeros_like(grad_acc),
                                                use_locking=self._use_locking)
        return tf.group(adam_op, grad_acc_to_zero_op)
        
      ## accumulate gradients for var
      def _accumulate_gradient(grad_acc, grad):
        assign_op = tf.assign_add(grad_acc, grad, use_locking=self._use_locking)     
        return tf.group(assign_op) 
      ##apply adam or accumulate gradients for 'Var'
      update_op = tf.cond(tf.equal(self._get_iter_variable(), 0),
                   lambda: _apply_adam(grad_acc, grad, param, m, v),
                   lambda: _accumulate_gradient(grad_acc, grad))
      update_ops.append(update_op)

    ##do extra update ops for some var
    apply_updates = self._finish(update_ops, name_scope=name)      
    return apply_updates

  def _finish(self, update_ops, name_scope):
    """
    iter <- iter + 1 mod n
    """
    iter_ = self._get_iter_variable()
    with tf.control_dependencies(update_ops):
      with tf.colocate_with(iter_):
        update_iter = iter_.assign(tf.mod(iter_ + 1, self._n_t),
                                    use_locking=self._use_locking)
    return tf.group(
        *update_ops+[update_iter], name=name_scope)  

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
