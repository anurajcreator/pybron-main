from datetime import date,datetime
from flask import request
from typing import List, Dict  # noqa: F401


def apiresponse(success = "" ,message = "" ,error = 'null',data =  'null' ):

    return {
    "success": success,
    "message": message,
    "data": data,
    "error": error,
}


def date_to_str(date):
    if date:
        return str(date.day) + "-" + str(date.month) + "-" + str(date.year)
    else:
        return ""

def str_to_date(data):
    n_date = data.split("-")
    data = datetime(int(n_date[0]),int(n_date[1]),int(n_date[2]))
    return data


class ApiResponse:
    def __init__(self, success: str = None, message: str = None, data: List[object] = None, error: str = None):
        self.success = success
        self.message = message
        self.data = data
        self.error = error

    # @classmethod
    # def from_dict(cls, dikt) -> 'ApiResponse':
    #     """Returns the dict as a model

    #     :param dikt: A dict.
    #     :type: dict
    #     :return: The ApiResponse of this ApiResponse.  # noqa: E501
    #     :rtype: ApiResponse
    #     """
    #     return util_file.deserialize_model(dikt, cls)

    # @property
    # def success(self) -> str:
    #     """Gets the type of this ApiResponse.

    #     :return: The type of this ApiResponse.
    #     :rtype: str
    #     """
    #     return self._success

    # @success.setter
    # def success(self, success: str):
    #     """Sets the type of this ApiResponse.

    #     :param type: The type of this ApiResponse.
    #     :type type: str
    #     """
    #     self._success = success

    # @property
    # def message(self) -> str:
    #     """Gets the message of this ApiResponse.

    #     :return: The message of this ApiResponse.
    #     :rtype: str
    #     """
    #     return self._message

    # @message.setter
    # def message(self, message: str):
    #     """Sets the message of this ApiResponse.

    #     :param message: The message of this ApiResponse.
    #     :type message: str
    #     """

    #     self._message = message

    # @property
    # def error(self) -> str:
    #     """Gets the description of this ApiResponse.

    #     :return: The description of this ApiResponse.
    #     :rtype: str
    #     """
    #     return self._description

    # @error.setter
    # def error(self, error: str):
    #     """Sets the description of this ApiResponse.

    #     :param description: The description of this ApiResponse.
    #     :type description: str
    #     """

    #     self._error = error

    # @property
    # def data(self) -> List[object]:
    #     """Gets the data of this ApiResponse.

    #     :return: The data of this ApiResponse.
    #     :rtype: List[object]
    #     """
    #     return self._data

    # @data.setter
    # def data(self, data: List[object]):
    #     """Sets the data of this ApiResponse.

    #     :param data: The data of this ApiResponse.
    #     :type data: List[object]
    #     """

    #     self._data = data


    
 
