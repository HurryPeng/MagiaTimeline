import typing
import enum

class AbstractFlagIndex(enum.IntEnum):
    # concrete method
    @classmethod
    def getNum(cls) -> int:
        return len(cls.__members__)

    # concrete method
    @classmethod
    def getDefaultFlags(cls) -> typing.List[typing.Any]:
        # returns a list of default values corresponding to each flag type
        # adds a None for index 0 because valid flags are numbered from 1
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
