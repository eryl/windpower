import datetime

def timestamp():
    """
    Generates a timestamp.
    :return:
    """
    t = datetime.datetime.now().replace(microsecond=0)
    #Since the timestamp is usually used in filenames, isoformat will be invalid in windows.
    #return t.isoformat()
    # We'll use another symbol instead of the colon in the ISO format
    # YYYY-MM-DDTHH:MM:SS -> YYYY-MM-DDTHH.MM.SS
    time_format = "%Y-%m-%dT%H.%M.%S"
    return t.strftime(time_format)

