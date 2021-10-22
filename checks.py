"""I don't like how the error messages are shown in attrs"""

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