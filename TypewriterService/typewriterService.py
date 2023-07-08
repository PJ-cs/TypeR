import serial
import serial.tools.list_ports as list_ports
import numpy as np

def find_arduino(serial_number: str) -> serial.Serial:
    for pinfo in serial.tools.list_ports.comports():
        if pinfo.serial_number == serial_number:
            return serial.Serial(pinfo.device)
    raise IOError(f"Could not find the arduino {serial_number} - is it plugged in?")

SERIAL_NUMBER : str = "550373133373515140E1"
LETTER_LIST: list[str] = [
    ".", ">", "%o", "<", "|", "'", "³", "_", "Y", "J", "?", "X", "Ö", "V", "B", "W",
    "P", "Q", "H", "S", "A", "I", "O", "E", "U", "L", "R", "T", "C", "N", "G", "D",
    "F", "M", "Z", "K", "Ä", "Ü", ";", "!", "\"", "§", "$", "=", "%", "&", "/", "(",
    ")", "`", "*", "°", ":", "#", "²", "-", "y", "j", "ß", "x", "ö",
    "v", "d", "w", "p", "q", "h", "s", "a", "i", "o", "e", "u", "l", "r", "t", "c",
    "n", "g", "b", "f", "m", "z", "k", "ä", "ü", ",", "1", "2", "3", "4", "0", "5",
    "6", "7", "8", "9", "´", "+", "µ"
 
]

arduino: serial.Serial = find_arduino(SERIAL_NUMBER)
def write_letter(x: int, y: int, letter: int, thickness: int) -> str:
    command : str = f"<X{x} Y{y} L{letter} T{thickness}>"
    arduino.write(bytes(command, 'utf-8'))
    return arduino.read()


def write_img(img: np.ndarray):
    for row_index, row in enumerate(img):
        for column_index, pixel in enumerate(row):
            # X for typewriter is movement in line, pixel[0] is letter index according to list above
            # pixel[1] is thickness 
            reponse_code = write_letter(column_index, row_index, pixel[0], pixel[1])
            if reponse_code =="R":
                # TODO failure routine, to continue here when restarted
                raise IOError("The typewriter ribbon is empty")
            elif reponse_code != "A":
                raise Exception(f"Encountered unexpected arduino response: '{reponse_code}'")

# ham levels start with 1
print(arduino.readline())

for x in range(0, 40):
    for y in range(0, 40):
        if write_letter(x, y, 0, 3) != bytes('A', "UTF-8"):
            exit(1)