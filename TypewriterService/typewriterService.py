import serial
import serial.tools.list_ports as list_ports
import numpy as np
import time
import random


def find_arduino(serial_number: str) -> serial.Serial:
    for pinfo in serial.tools.list_ports.comports():
        if pinfo.serial_number == serial_number:
            arduinoTmp = serial.Serial(pinfo.device)
            print(arduinoTmp.readline())
            return arduinoTmp
    raise IOError(f"Could not find the arduino {serial_number} - is it plugged in?")


SERIAL_NUMBER: str = "550373133373515140E1"
HORIZONTAL_PIXEL_PER_LETTER: int = 20
VERTICAL_PIXEL_PER_LETTER: int = 40
HORIZONTAL_LIMIT = 1200  # TODO
LETTER_LIST: list[str] = [
    ".",
    ">",
    "‰",
    "<",
    "|",
    "'",
    "³",
    "_",
    "Y",
    "J",
    "?",
    "X",
    "Ö",
    "V",
    "B",
    "W",
    "P",
    "Q",
    "H",
    "S",
    "A",
    "I",
    "O",
    "E",
    "U",
    "L",
    "R",
    "T",
    "C",
    "N",
    "G",
    "D",
    "F",
    "M",
    "Z",
    "K",
    "Ä",
    "Ü",
    ";",
    "!",
    '"',
    "§",
    "$",
    "=",
    "%",
    "&",
    "/",
    "(",
    ")",
    "`",
    "*",
    "°",
    ":",
    "#",
    "²",
    "-",
    "y",
    "j",
    "ß",
    "x",
    "ö",
    "v",
    "d",
    "w",
    "p",
    "q",
    "h",
    "s",
    "a",
    "i",
    "o",
    "e",
    "u",
    "l",
    "r",
    "t",
    "c",
    "n",
    "g",
    "b",
    "f",
    "m",
    "z",
    "k",
    "ä",
    "ü",
    ",",
    "1",
    "2",
    "3",
    "4",
    "0",
    "5",
    "6",
    "7",
    "8",
    "9",
    "´",
    "+",
    "µ",
]
LETTER_DICT: dict[str, int] = {letter: i for i, letter in enumerate(LETTER_LIST)}

arduino: serial.Serial = find_arduino(SERIAL_NUMBER)

# TODO create typewriter object with methods, constraints (rows, columns), information
def write_letter(x: int, y: int, letter: int, thickness: int) -> bytes:
    command: str = f"<X{int(x)} Y{int(y)} L{int(letter)} T{int(thickness)}>"
    arduino.write(bytes(command, "utf-8"))
    handleArduinoReturn(arduino.readline())


# TODO this is wrong, there can be multiple letters per pixel, up to five channels
def write_img(np_letter: np.ndarray, np_strength: np.ndarray):
    assert(np.all(np_letter.shape == np_strength.shape))
    letter_per_pix, height, width = np_letter.shape
    for row_index in range(height):
        for column_index in range(width):
            # X for typewriter is movement in line, pixel[0] is letter index according to list above
            # pixel[1] is thickness
            for channel in range(letter_per_pix):
                strength = np_strength[channel][row_index][column_index]
                if strength > 0:
                    letter_index = np_letter[channel][row_index][column_index]
                    response_code = write_letter(column_index, row_index, letter_index, strength)
                    handleArduinoReturn(response_code)


def handleArduinoReturn(response_code: bytes):
    response_str = response_code.decode("utf-8")
    if response_str.startswith("R"):
        # TODO failure routine, to continue here when restarted
        raise IOError("The typewriter ribbon is empty")
    elif not response_str.startswith("A"):
        raise Exception(f"Encountered unexpected arduino response: '{response_code}'")


def write_text(text: list[str], thickness: int):
    # print("writing text: ", text)
    running_horizontal = 0
    running_vertical = 0

    for letter in text:
        if letter == " ":
            if running_horizontal != 0:
                running_horizontal += HORIZONTAL_PIXEL_PER_LETTER
            continue
        elif letter == "\n":
            running_vertical += VERTICAL_PIXEL_PER_LETTER
            running_horizontal = 0
            continue
        else:
            if letter not in LETTER_DICT:  # print space, if letter unknown
                running_horizontal += HORIZONTAL_PIXEL_PER_LETTER
                continue
            handleArduinoReturn(
                write_letter(
                    running_horizontal, running_vertical, LETTER_DICT[letter], thickness
                )
            )
        running_horizontal += HORIZONTAL_PIXEL_PER_LETTER

        if running_horizontal > HORIZONTAL_LIMIT:
            running_horizontal = 0
            running_vertical += VERTICAL_PIXEL_PER_LETTER


def write_letter_sample():
    print_string = []
    for letter in LETTER_LIST:
        print_string.append(letter)
        print_string.append(" ")
    write_text(print_string, 1.0)


def write_thickness_test():
    for index in range(0, 100):
        for x_pos in range(
            0, HORIZONTAL_PIXEL_PER_LETTER * 30, HORIZONTAL_PIXEL_PER_LETTER
        ):
            thickness = x_pos // HORIZONTAL_PIXEL_PER_LETTER * 4
            write_letter(
                x_pos + index // 33 * HORIZONTAL_PIXEL_PER_LETTER * 30,
                index % 33 * VERTICAL_PIXEL_PER_LETTER,
                index,
                thickness,
            )


def write_accuracy_test():
    for sample_id in range(10):
        random.seed(42)
        offset_horizontal = sample_id * HORIZONTAL_PIXEL_PER_LETTER
        for i in range(10):
            pos_abs = 1.8**i
            letter = random.randint(0, 99)
            thickness = random.randint(10, 255)
            print(f"letter: {LETTER_LIST[letter]} letter ind: {letter} thickness: {thickness}")
            write_letter(
                offset_horizontal + pos_abs,
                pos_abs,
                letter,
                thickness,
            )


time.sleep(2)
# write_letter_sample()
# write_text([".", "*", ".","*", ".","*", ".","*", "."], 1)

# write_text(list("Guten Morgen, Maria!"), 0.1)

# for i in range(0, 10):
#     for j in range(0, 10):
#         write_letter(i*10, j*10, 0, 1)

#write_thickness_test()

write_accuracy_test()
