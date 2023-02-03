from __future__ import annotations
import typing
import enum

class AbstractFlagIndex(enum.IntEnum):
    # similar to setting Debug = 0 in this enum class
    # but does not forbid this abstract class from being inherited
    # concrete method
    @classmethod
    @property
    def Debug(cls) -> int:
        return 0

    # concrete method
    @classmethod
    def getNum(cls) -> int:
        return len(cls.__members__)

    # concrete method
    @classmethod
    def getDefaultFlags(cls) -> typing.List[typing.Any]:
        # returns a list of default values corresponding to each flag type
        # valid flags are numbered from 1, and position 0 is reserved for debug information
        flags = cls.getDefaultFlagsImpl()
        if not len(flags) == cls.getNum():
            raise Exception("The length of default values must be equal to the number of flags")
        return [None] + flags

    # abstract method
    @classmethod
    def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
        # returns a list of default values corresponding to each flag type
        # the length of this list must be equal to the number of flags
        return []
