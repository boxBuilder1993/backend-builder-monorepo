"""Entity base class.

An entity class is the key under which its rows are stored. For now rows are
plain dicts and the class carries no fields beyond the system ones; it exists
so that every store and context operation is addressed by a type rather than
a string.
"""


class Entity:
    """Base class for all entities. Subclass to declare a new entity.

    Every row carries ``id``, a UUID string assigned by the store on create.
    """

    id: str
