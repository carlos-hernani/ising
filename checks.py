"""I don't like how the error messages are shown in attrs"""

import attr
import numpy as np
from attr._make import attrib, attrs


@attrs(repr=False, slots=True, hash=True)
class _InstanceOfValidator(object):
    type = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        Changed the format to something more compact.
        """
        if not isinstance(value, self.type):
            raise TypeError(f"'{attr.name}' must be {self.type}")

    def __repr__(self):
        return "<instance_of validator for type {type!r}>".format(
            type=self.type
        )


def instance_of(type):
    """
    A validator that raises a `TypeError` if the initializer is called
    with a wrong type for this particular attribute (checks are performed using
    `isinstance` therefore it's also valid to pass a tuple of types).
    :param type: The type to check for.
    :type type: type or tuple of types
    :raises TypeError: With a human readable error message, the attribute
        (of type `attr.Attribute`), the expected type, and the value it
        got.
    """
    return _InstanceOfValidator(type)


# Validators: They check the inputs.

def gtzero(instance, attribute, value):
    """
    gtzero Validator: checks greather than zero
    """    
    if value <= 0:
        raise ValueError(f'{attribute.name} must be positive & non-zero.')

def gele(instance, attribute, value):
    """
    gele Validator: checks geq than zero or leq than one
    """    
    if value < 0 or value > 1:
        raise ValueError(f'{attribute.name} must be between [0,1].')


def opt_type(type, cond=None, default_value=None):
    """
    opt_type Enforces Optional Type and validates conditions.

    Args:
        type ([type]): The desired type
        cond (callable, optional): Condition function. Defaults to None.
        default_value ([type], optional): The default value. Defaults to None.

    Returns:
        dict: unpack it in attr.ib
    """    
    ret_value = {
            'validator': [attr.validators.optional(
                instance_of(type))
                ],
            'default': default_value}
    if cond is not None:
        ret_value['validator'] = [
            attr.validators.optional(
                instance_of(type)
                ),
            cond
        ]
    
    return ret_value