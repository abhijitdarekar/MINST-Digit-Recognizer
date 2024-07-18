import sys

def ErrorMessageDetail(error, errorDetail:sys):
    _,_, error_tb = errorDetail.exc_info()
    fileName = error_tb.tb_frame.f_code.co_filename
    errorMessage = " Error Occured in Python Scrpt [{0}]| Line Number [{1}] | Message [{2}]".format(
        fileName, error_tb.tb_lineno, str(error)
    )
    return errorMessage

class CustomException(Exception):
    def __init__(self, errorMessage, errorDetail:sys):
        super().__init__(errorMessage)
        self.errorMessage = ErrorMessageDetail(errorMessage,errorDetail)

    def __str__(self) -> str:
        return self.errorMessage