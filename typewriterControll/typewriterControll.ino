//https://hackaday.io/project/183713-using-the-arduino-accelstepper-library
# include <AccelStepper.h>

// Stepper State definitions
#define RUNNING 01
#define WAITING 02
#define STOPPED 03
#define AT_POS 04
#define RUN_HOME 05

//TODO make sure jumpers are set
//TODO experiment with MAX_SPEED docu says less than 1000
//TODO experiment with ACCELeration
//TODO experiment with MAX_STEPS
//TODO experiment with STEPS_PER_ROT
//TODO experiment with HOMING_SPEED -/+ ?
//TODO experiment with START_OFFSET_X

// ink ribbon movement
const byte DIR_PIN_A = 13; 
const byte STEP_PIN_A = 12;
const float MAX_SPEED_A = 1000; 
const float ACCEL_A = 50;
const float MAX_POS = 10000;
const int MAX_STEPS_A = 10000;
int stateA;
float currentAGoal;
float nextAGoal;

// carriage, horizontal movement
const byte DIR_PIN_X = 5;
const byte STEP_PIN_X = 2;
const float MAX_SPEED_X = 1000;
const float HOMING_SPEED_X = MAX_SPEED_X /2;
// how many steps to the right of stop
// switch to take, where new home pos
const float START_OFFSET_X = 100;
const float ACCEL_X = 50;
const int MAX_STEPS_X = 10000;
int stateX;
float currentXGoal;
float nextXGoal;

// line feed, vertical movement
const byte DIR_PIN_Y = 6;
const byte STEP_PIN_Y = 3;
const float MAX_SPEED_Y = 1000;
const float ACCEL_Y = 50;
// sheet height 
const int MAX_STEPS_Y = 10000;
int stateY;
float currentYGoal;
float nextYGoal;

// daisy wheel
const byte DIR_PIN_Z = 7;
const byte STEP_PIN_Z = 4;
const float MAX_SPEED_Z = 1000;
const float HOMING_SPEED_Z = MAX_SPEED_Z;
//steps for one full rotation
const int MAX_STEPS_Z = 10000;
const float ACCEL_Z = 50;
int stateZ;
float currentZGoal;
float nextZGoal;

const byte LIMIT_X_AXIS_PIN = 9; 

const byte STEPPER_ENABLE_PIN = 8;

const byte HAMMER_PIN = 11;

AccelStepper stepperA(AccelStepper::DRIVER, STEP_PIN_A, DIR_PIN_A);
AccelStepper stepperX(AccelStepper::DRIVER, STEP_PIN_X, DIR_PIN_X);
AccelStepper stepperY(AccelStepper::DRIVER, STEP_PIN_Y, DIR_PIN_Y);
AccelStepper stepperZ(AccelStepper::DRIVER, STEP_PIN_Z, DIR_PIN_Z);

const byte numChars = 32;
char receivedCommand[numChars];

boolean newCommand = false;
boolean startUp = true;

void setup() {
  // put your setup code here, to run once:
  //TODO check if needs to be inverted
  pinMode(STEPPER_ENABLE_PIN, OUTPUT);
  digitalWrite(STEPPER_ENABLE_PIN, HIGH);

  pinMode(HAMMER_PIN, OUTPUT);
  analogWrite(HAMMER_PIN, 0);

  pinMode(LIMIT_X_AXIS_PIN, INPUT);

  stepperA.setMaxSpeed(MAX_SPEED_A);
  stepperA.setAcceleration(ACCEL_A);
  stateA = WAITING;

  stepperX.setMaxSpeed(MAX_SPEED_X);
  stepperX.setAcceleration(ACCEL_X);
  stateX = WAITING;

  stepperY.setMaxSpeed(MAX_SPEED_Y);
  stepperY.setAcceleration(ACCEL_Y);
  stateY = WAITING;

  stepperZ.setMaxSpeed(MAX_SPEED_Z);
  stepperZ.setAcceleration(ACCEL_Z);
  stateZ = WAITING;


  Serial.begin(9600);
  Serial.println("<Arduino is ready>");
}

void loop() {
  doStartUp(startUp);
  recvCommand();
  processNewCommand(); 

}

void recvCommand() {
  static boolean recvInProgress = false;
  static byte ndx = 0;
  char startMarker = '<';
  char endMarker = '>';
  char rc;

  while (Serial.available() > 0 && newCommand == false) {
      rc = Serial.read();

      if (recvInProgress == true) {
          if (rc != endMarker) {
              receivedCommand[ndx] = rc;
              ndx++;
              if (ndx >= numChars) {
                  ndx = numChars - 1;
              }
          }
          else {
              receivedCommand[ndx] = '\0'; // terminate the string
              recvInProgress = false;
              ndx = 0;
              newCommand = true;
          }
      }

      else if (rc == startMarker) {
          recvInProgress = true;
      }
  }
}

void processNewCommand() {
  if (newCommand == true) {
    char* commandTmp = strtok(receivedCommand, " ");
    while (commandTmp != 0){
      
    }

      
    //TODO
    // command has format: X Y L T
    // X/Y is position on X/Y-Axis 
    // L is the Letter to print
    // T is Thickness of the letter
    newCommand = false;
  }
}

void confirmCommandRecieved() {
  Serial.write("ACK\n");
  }

void allAreWaiting() {
  return stateA == WAITING && stateX == WAITING && stateY == WAITING && stateZ == WAITING;
}

void doStartUp(bool startUpParam){
  if(startUpParam){
    // move daisy wheel and horizontal movement to home position
    switch(stateX){
      case WAITING:
        stepperX.setSpeed(HOMING_SPEED_X);
        stateX = RUNNING;
        break;
      case RUNNING:
        if(digitalRead(LIMIT_X_AXIS_PIN) == HIGH){
          stepperX.stop();
          stateX = AT_POS;
          break;
        }
        stepperX.runSpeed();
        break;
      case AT_POS:
        if(stateZ == AT_POS){
          stepperX.move(START_OFFSET_X);
          stateX = RUN_HOME;
        }
        break;
      case RUN_HOME:
        if(stepperX.distanceToGo() == 0){
          startUp = false;
          stateX = WAITING;
          stateZ = WAITING;
          stepperX.setCurrentPosition(0);
        }        
        break;
      default:
        break;
    }
    
    switch(stateZ){
      case WAITING:
        if(stateX == AT_POS){
          // if horizontal at stop switch, try rotate daisy wheel two times,
          // will be physically blocked at 0 pos, then reset
          stepperZ.moveTo(MAX_STEPS_Z*2); 
          stateZ = RUNNING;
        }
        break;
      case RUNNING:
        if(stepperZ.distanceToGo() == 0){
          stepperZ.setCurrentPosition(0);
          stateZ = AT_POS;
          break;
        }
        stepperZ.run();
        break;
      default:
        break;
    }
  }
}
