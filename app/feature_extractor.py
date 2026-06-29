import pyshark


def read_capture(file_path):

    capture = pyshark.FileCapture(
        file_path,
        tshark_path=r"C:\Program Files\Wireshark\tshark.exe"
    )

    return capture