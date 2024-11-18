# pyhidapi is used to read inputs from the PS4 controller.
# pip install pyhidapi
# The PS4 controller should be connected via USB
# So far only tested on Mac

import math
import logging
import threading
import hid
import time
import struct

class PS4JoystickController:
    def __init__(
        self,
        motor_names,
        initial_position=[90, 170, 170, 0, 0, 10],
        l1=117.0,  # Length of first lever in mm
        l2=136.0,  # Length of second lever in mm
        *args,
        **kwargs
    ):
        self.motor_names = motor_names
        self.initial_position = initial_position
        self.current_positions = {name: pos for name, pos in zip(self.motor_names, self.initial_position)}
        self.new_positions = self.current_positions.copy()

        # Inverse Kinematics parameters are used to compute x and y positions
        self.l1 = l1
        self.l2 = l2

        # x and y are coordinates of the axis of wrist_flex motor relative to the axis of the shoulder_pan motor in mm
        self.x, self.y = self._compute_position(self.current_positions['shoulder_lift'], self.current_positions['elbow_flex'])

        # Gamepad states
        self.axes = {
            'RX': 0.0,
            'RY': 0.0,
            'LX': 0.0,
            'LY': 0.0,
            'L2': 0.0,
            'R2': 0.0,
        }
        self.buttons = {
            'L2': 0,
            'R2': 0,
            'DPAD_LEFT': 0,
            'DPAD_RIGHT': 0,
            'DPAD_UP': 0,
            'DPAD_DOWN': 0,
            'X': 0,
            'O': 0,
            'T': 0,
            'S': 0,
            'L1': 0,
            'R1': 0,
            'SHARE': 0,
            'OPTIONS': 0,
            'PS': 0,
            'L3': 0,
            'R3': 0,
        }

        # PS4 Controller constants
        self.VENDOR_ID = 0x054C  # Sony
        self.PRODUCT_ID = 0x09CC  # DualShock 4 Wireless Controller

        # D-Pad Direction Mapping
        self.DPAD_DIRECTIONS = {
            0x00: "DPAD_UP",
            0x01: "DPAD_UP_RIGHT",
            0x02: "DPAD_RIGHT",
            0x03: "DPAD_DOWN_RIGHT",
            0x04: "DPAD_DOWN",
            0x05: "DPAD_DOWN_LEFT",
            0x06: "DPAD_LEFT",
            0x07: "DPAD_UP_LEFT",
            0x08: "DPAD_NEUTRAL",
            0x0F: "DPAD_NEUTRAL",
        }

        # Face Buttons Mapping (Bits 4-7 of Byte 5)
        self.FACE_BUTTONS = {
            0x10: "S",
            0x20: "X",
            0x40: "O",
            0x80: "T",
        }

        # Additional Buttons (Byte 6)
        self.ADDITIONAL_BUTTONS = {
            0x01: "L1",
            0x02: "R1",
            0x04: "L2",
            0x08: "R2",
            0x10: "SHARE",
            0x20: "OPTIONS",
            0x40: "L3",
            0x80: "R3",
        }

        # Special Buttons (Byte 7)
        self.SPECIAL_BUTTONS = {
            0x01: "PS",
            0x02: "TOUCHPAD",
        }

        # Initialize hid device
        self.device = None
        self.running = False
        self.connect()

        # Start the thread to read inputs
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

    def connect(self):
        try:
            self.device = hid.device()
            self.device.open(self.VENDOR_ID, self.PRODUCT_ID)
            logging.info(f"Connected to {self.device.get_manufacturer_string()} {self.device.get_product_string()}")
            self.running = True
        except IOError as e:
            logging.error(f"Unable to open device: {e}")
            self.device = None

    def disconnect(self):
        self.running = False
        if self.device:
            self.device.close()
            logging.info("Controller disconnected.")
            self.device = None

    def read_loop(self):
        while self.running:
            try:
                data = self.device.read(64, timeout_ms=100)
                if data:
                    self._process_gemepad_input(data)
            except Exception as e:
                logging.error(f"Error reading from device: {e}")
                time.sleep(1)  # Wait before retrying
                self.connect()

    def _process_gemepad_input(self, data):
        with self.lock:
            # Byte 1-4: Analog sticks
            left_stick_x = data[1] - 128
            left_stick_y = data[2] - 128
            right_stick_x = data[3] - 128
            right_stick_y = data[4] - 128

            # Normalize to -1.0 to 1.0
            self.axes['LX'] = self._filter_deadzone(left_stick_x / 128.0)
            self.axes['LY'] = self._filter_deadzone(-left_stick_y / 128.0)
            self.axes['RX'] = self._filter_deadzone(right_stick_x / 128.0)
            self.axes['RY'] = self._filter_deadzone(-right_stick_y / 128.0)

            # Byte 5: D-Pad and face buttons
            buttons_byte = data[5]

            # D-Pad is lower 4 bits
            dpad_bits = buttons_byte & 0x0F
            dpad_direction = self.DPAD_DIRECTIONS.get(dpad_bits, "DPAD_NEUTRAL")

            # Reset D-Pad buttons
            self.buttons['DPAD_UP'] = 0
            self.buttons['DPAD_DOWN'] = 0
            self.buttons['DPAD_LEFT'] = 0
            self.buttons['DPAD_RIGHT'] = 0

            if "UP" in dpad_direction:
                self.buttons['DPAD_UP'] = 1
            if "DOWN" in dpad_direction:
                self.buttons['DPAD_DOWN'] = 1
            if "LEFT" in dpad_direction:
                self.buttons['DPAD_LEFT'] = 1
            if "RIGHT" in dpad_direction:
                self.buttons['DPAD_RIGHT'] = 1

            # Face buttons are bits 4-7
            face_buttons_bits = buttons_byte & 0xF0
            for bitmask, name in self.FACE_BUTTONS.items():
                self.buttons[name] = 1 if face_buttons_bits & bitmask else 0

            # Byte 6: Additional buttons
            buttons_byte2 = data[6]
            for bitmask, name in self.ADDITIONAL_BUTTONS.items():
                self.buttons[name] = 1 if buttons_byte2 & bitmask else 0

            # Byte 7: Special buttons
            special_buttons_byte = data[7]
            for bitmask, name in self.SPECIAL_BUTTONS.items():
                self.buttons[name] = 1 if special_buttons_byte & bitmask else 0

            # Byte 8: L2 Analog
            l2_analog = data[8]
            self.axes['L2'] = l2_analog / 255.0  # 0.0 to 1.0

            # Byte 9: R2 Analog
            r2_analog = data[9]
            self.axes['R2'] = r2_analog / 255.0  # 0.0 to 1.0

            axes=self.axes.copy()
            buttons=self.buttons.copy()

        self._update_positions(axes, buttons)

    def _filter_deadzone(self, value, threshold=0.1):
        """
        Apply a deadzone to the joystick input to avoid drift.
        """
        if abs(value) < threshold:
            return 0.0
        return value

    def get_command(self):
        """
        Return the current motor positions after reading and processing inputs.
        """
        return self.current_positions.copy()

    def _update_positions(self, axes, buttons):
        # Compute new positions based on inputs
        SPEED = 0.3
        # TODO: speed can be different for different directions

        temp_positions = self.current_positions.copy()

        # Handle macro buttons
        # Buttons have assigned states where robot can move directly
        used_macros = False
        macro_buttons = ['X', 'O', 'T', 'S'] # can use more if needed
        for button in macro_buttons:
            if buttons.get(button):
                temp_positions = self._execute_macro(button, temp_positions)
                temp_x, temp_y = self._compute_position(temp_positions['shoulder_lift'], temp_positions['elbow_flex'])
                correct_inverse_kinematics = True
                used_macros = True

        if not used_macros:
            # Map joystick inputs to motor positions
            # Right joystick controls "wrist_roll" (left/right) and "wrist_flex" (up/down)
            temp_positions['wrist_roll'] += axes['RX'] * SPEED  # degrees per update
            temp_positions['wrist_flex'] -= axes['RY'] * SPEED  # degrees per update

            # L2 and R2 control gripper
            temp_positions['gripper'] -= SPEED * axes['R2']  # Close gripper
            temp_positions['gripper'] += SPEED * axes['L2'] # Open gripper

            # Left joystick and dpad left and right control shoulder_pan
            temp_positions['shoulder_pan'] += (axes['LX'] - buttons['DPAD_LEFT'] + buttons['DPAD_RIGHT']) * SPEED  # degrees per update

            # Handle the linear movement of the arm
            # Left joystick up/down changes x
            temp_x = self.x + axes['LY'] * SPEED  # mm per update

            # D-pad up/down change y
            temp_y = self.y + (buttons['DPAD_UP'] - buttons['DPAD_DOWN']) * SPEED

            correct_inverse_kinematics = False

            # Compute shoulder_lift and elbow_flex angles based on x and y
            try:
                temp_positions['shoulder_lift'], temp_positions['elbow_flex'] = self._compute_inverse_kinematics(temp_x, temp_y)
                shoulder_lift_change = temp_positions['shoulder_lift'] - self.current_positions['shoulder_lift']
                elbow_flex_change = temp_positions['elbow_flex'] - self.current_positions['elbow_flex']
                temp_positions['wrist_flex'] +=  shoulder_lift_change - elbow_flex_change
                correct_inverse_kinematics = True
            except ValueError as e:
                logging.error(f"Error computing inverse kinematics: {e}")

        # Perform eligibility check
        if self._is_position_valid(temp_positions, temp_x, temp_y) and correct_inverse_kinematics:
            # Atomic update: all positions are valid, apply the changes
            self.current_positions = temp_positions
            self.x = temp_x
            self.y = temp_y
        else:
            # Invalid positions detected, do not update
            logging.warning("Invalid motor positions detected. Changes have been discarded.")
            # TODO: reimplement rumble
            # self.rumble(0.5, 0.5, 200)

    def _is_position_valid(self, positions, x, y):
        """
        Check if all positions are within their allowed ranges.
        Define the allowed ranges for each motor.
        """
        allowed_ranges = {
            'shoulder_pan': (-10, 190),
            'shoulder_lift': (-5, 185),
            'elbow_flex': (-5, 185),
            'wrist_flex': (-110, 110),
            'wrist_roll': (-110, 110),
            'gripper': (0, 100),
            'x': (15, 250),
            'y': (-110, 250),
        }

        for motor, (min_val, max_val) in allowed_ranges.items():
            if motor in positions:
                if not (min_val <= positions[motor] <= max_val):
                    logging.error(f"Motor '{motor}' position {positions[motor]} out of range [{min_val}, {max_val}].")
                    return False
                
        # Check if x and y positions are within the allowed ranges
        if x < allowed_ranges['x'][0] or x > allowed_ranges['x'][1]:
            logging.error(f"X position {x} out of range {allowed_ranges['x']}.")
            return False
    
        if y < allowed_ranges['y'][0] or y > allowed_ranges['y'][1]:
            logging.error(f"Y position {y} out of range {allowed_ranges['y']}.")
            return False

        return True

    def _execute_macro(self, button, positions):
        """
        Define macros for specific buttons. When a macro button is pressed,
        set the motors to predefined positions.
        """
        macros = {
            'O': [90, 180, 180, 0, 0, 10], # initial position
            'X': [90, 50, 130, -90, 90, 80], # low horizontal gripper
            'T': [90, 130, 150, 70, 90, 80], # top down gripper
            'S': [90, 160, 140, 20, 0, 0], # looking forward
            # can add more macros for all other buttons
        }

        if button in macros:
            motor_positions = macros[button][:6]
            for name, pos in zip(self.motor_names, motor_positions):
                positions[name] = pos
            logging.info(f"Macro '{button}' executed. Motors set to {motor_positions}.")
        return positions

    def _compute_inverse_kinematics(self, x, y):
        """
        Compute motor 2 and motor 3 angles based on the desired x and y positions.
        """
        #TODO: add explanation of the math behind this
        #TODO: maybe the math can be optimized, check it

        l1 = self.l1
        l2 = self.l2

        # Compute the distance from motor 2 to the desired point
        distance = math.hypot(x, y)

        # Check if the point is reachable
        if distance > (l1 + l2) or distance < abs(l1 - l2):
            raise ValueError(f"Point ({x}, {y}) is out of reach.")

        # Compute angle for motor3 (theta2)
        cos_theta2 = (l1**2 + l2**2 - distance**2) / (2 * l1 * l2)
        # Clamp the value to avoid numerical issues
        cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
        theta2_rad = math.acos(cos_theta2)
        theta2_deg = math.degrees(theta2_rad)
        # Adjust motor3 angle

        offset = math.degrees(math.asin(32/l1))

        motor3_angle = 180 - (theta2_deg - offset)

        # Compute angle for motor2 (theta1)
        cos_theta1 = (l1**2 + distance**2 - l2**2) / (2 * l1 * distance)
        # Clamp the value to avoid numerical issues
        cos_theta1 = max(min(cos_theta1, 1.0), -1.0)
        theta1_rad = math.acos(cos_theta1)
        theta1_deg = math.degrees(theta1_rad)
        alpha_rad = math.atan2(y, x)
        alpha_deg = math.degrees(alpha_rad)

        beta_deg = 180 - alpha_deg - theta1_deg
        motor2_angle = 180 - beta_deg + offset

        return motor2_angle, motor3_angle

    def _compute_position(self, motor2_angle, motor3_angle):
        """
        Compute the x and y positions based on the motor 2 and motor 3 angles.
        """
        l1 = self.l1
        l2 = self.l2
        offset = math.degrees(math.asin(32/l1))

        beta_deg = 180 - motor2_angle + offset
        beta_rad = math.radians(beta_deg)

        theta2_deg = 180 - motor3_angle + offset
        theta2_rad = math.radians(theta2_deg)

        y = l1 * math.sin(beta_rad) - l2 * math.sin(beta_rad - theta2_rad)
        x = - l1 * math.cos(beta_rad) + l2 * math.cos(beta_rad - theta2_rad)
        return x, y

    def stop(self):
        """
        Clean up resources.
        """
        self.disconnect()
        self.thread.join()
