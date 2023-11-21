import json
import serial
import numpy as np
import time
import random
import cv2
import serial.tools.list_ports

class Typewriter:
    def __init__(self, config_json_path: str, machine_name: str) -> None:
        with open(config_json_path, 'r') as f:
            config_dict = json.load(f)[machine_name]
            serial_number = config_dict["serialNumber"]
            self.letters_list = config_dict["letterList"]
            self.resolution_horiz = config_dict["resolutionHoriz"]
            self.resolution_vert = config_dict["resolutionVert"]
            self.pixel_per_letter_horiz = config_dict["pixelPerLetterHoriz"]
            self.pixel_per_letter_vert = config_dict["pixelPerLetterVert"]

            self.letter_dict: dict[str, int] = {letter: i for i, letter in enumerate(self.letters_list)}

            # TODO use the position in combination with resolution for error checks
            self.current_horizontal_pos = 0
            self.current_vertical_pos = 0
            self.current_letter_index = -1
            self.current_thickness = -1

            self.arduino: serial.Serial = self._find_arduino(serial_number)
            time.sleep(2)

    def _find_arduino(self, serial_number: str) -> serial.Serial:
        for pinfo in serial.tools.list_ports.comports():
            if pinfo.serial_number == serial_number:
                arduinoTmp = serial.Serial(pinfo.device)
                print(arduinoTmp.readline())
                return arduinoTmp
        raise IOError(f"Could not find the arduino {serial_number} - is it plugged in?")
    
    def _handleArduinoReturn(self, response_code: bytes):
        response_str = response_code.decode("utf-8")
        if response_str.startswith("R"):
            # TODO failure routine, to continue here when restarted, save state of typewrite to pickle, 
            raise IOError("The typewriter ribbon is empty")
        elif not response_str.startswith("A"):
            raise Exception(f"Encountered unexpected arduino response: '{response_code}'")

    def write_letter(self, x: int, y: int, letter: int, thickness: int) -> bytes:
        # assert correctness of input and limits
        assert(y >= 0 and y < self.resolution_vert)
        assert(x >= 0 and x < self.resolution_horiz)
        assert(letter >= 0 and letter < len(self.letters_list))
        assert(thickness >= 0 and thickness <= 255)
        self.current_horizontal_pos = x
        self.current_vertical_pos = y
        self.current_letter_index = letter
        self.current_thickness = thickness

        command: str = f"<X{int(x)} Y{int(y)} L{int(letter)} T{int(thickness)}>"
        self.arduino.write(bytes(command, "utf-8"))
        self._handleArduinoReturn(self.arduino.readline())
    
    def write_img(self, np_letter: np.ndarray, np_strength: np.ndarray):
        """
        np_letter array of letter indices [uint8]
        np_strength array of letter strengths [uint8]
        """
        assert(np.all(np_letter.shape == np_strength.shape))
        assert(len(np_letter.shape) == 3)
        letter_per_pix, height, width = np_letter.shape
        assert(height <= self.resolution_vert)
        assert(width <= self.resolution_horiz)
        for row_index in range(height):
            for column_index in range(width):
                # X for typewriter is movement in line, pixel[0] is letter index according to list above
                # pixel[1] is thickness
                for channel in range(letter_per_pix):
                    strength = int(np_strength[channel][row_index][column_index])
                    if strength > 0:
                        letter_index = np_letter[channel][row_index][column_index]
                        self.write_letter(column_index, row_index, letter_index, strength)

    def write_text(self, text: list[str], thickness: int):
        # print("writing text: ", text)
        running_horizontal = 0
        running_vertical = 0

        for letter in text:
            if letter == " ":
                if running_horizontal != 0:
                    running_horizontal += self.pixel_per_letter_horiz
                continue
            elif letter == "\n":
                running_vertical += self.pixel_per_letter_vert
                running_horizontal = 0
                continue
            else:
                assert(letter in self.letter_dict)
                self.write_letter(
                    running_horizontal, running_vertical, self.letter_dict[letter], thickness  
                )
            running_horizontal += self.pixel_per_letter_horiz

            if running_horizontal >= self.resolution_horiz:
                running_horizontal = 0
                running_vertical += self.pixel_per_letter_vert
    
    def write_letter_sample(self):
        print_string = []
        for letter in self.letters_list:
            print_string.append(letter)
            print_string.append(" ")
        self.write_text(print_string, 1.0)

    def write_thickness_test(self):
        for index in range(0, 100):
            for x_pos in range(
                0, self.pixel_per_letter_horiz * 30, self.pixel_per_letter_horiz
            ):
                thickness = x_pos // self.pixel_per_letter_horiz * 4
                self.write_letter(
                    x_pos + index // 33 * self.pixel_per_letter_horiz * 30,
                    index % 33 * self.pixel_per_letter_vert,
                    index,
                    thickness,
                )


    def write_accuracy_test(self):
        for sample_id in range(10):
            random.seed(42)
            offset_horizontal = sample_id * self.pixel_per_letter_horiz
            for i in range(10):
                pos_abs = 1.8**i
                letter_i = random.randint(0, 99)
                thickness = random.randint(10, 255)
                print(f"letter: {self.letters_list[letter_i]} letter ind: {letter_i} thickness: {thickness}")
                self.write_letter(
                    offset_horizontal + pos_abs,
                    pos_abs,
                    letter_i,
                    thickness,
                )


def bytes_2_np_img(bytes_img: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(bytes_img)
    np_img = cv2.imdecode(np_buffer, cv2.IMREAD_UNCHANGED)
    return np_img
