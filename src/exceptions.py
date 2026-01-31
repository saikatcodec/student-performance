import sys

def get_error(error, error_details: sys) -> str:
    _, _, traceback = error_details.exc_info()
    filename = traceback.tb_frame.f_code.co_filename
    error_message = f'The error occured in file name {filename} with line number {traceback.tb_lineno}. Error: {str(error)}'

    return error_message

class CustomException(Exception):
    def __init__(self, error, error_details: sys):
        super().__init__(error)
        self.error_details = get_error(error, error_details)

    def __str__(self):
        return self.error_details
