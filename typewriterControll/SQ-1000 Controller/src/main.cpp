#include <Arduino.h>
#include <AccelStepper.h>


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
//TODO experiment with START_OFFSET_X / Y
//TODO experiment with INCR_SIZE_A
//TODO replace onst with DEFINE for more efficiency and consistency

// ink ribbon movement
const byte DIR_PIN_A = 13; 
const byte STEP_PIN_A = 12;
const float MAX_SPEED_A = 10; 
const float ACCEL_A = 5;
//distance to move to have fresh ink ribbon
const float INCR_SIZE_A = 10;
const int MAX_STEPS_A = 10000;
int stateA;
//does not have goals, moves always same amount of steps

// carriage, horizontal movement
const byte DIR_PIN_X = 5;
const byte STEP_PIN_X = 2;
const float MAX_SPEED_X = 1000;
const float HOMING_SPEED_X = MAX_SPEED_X;
// how many steps to the right of stop
// switch to take, where new home pos
// if decrease OFFSET -> increase MAX steps by same amount
const float START_OFFSET_X = 400;
const float ACCEL_X = 10000;
const int MAX_STEPS_X = 17500;
int stateX;
int currentXGoal;
int nextXGoal;

// line feed, vertical movement
const byte DIR_PIN_Y = 6;
const byte STEP_PIN_Y = 3;
const float MAX_SPEED_Y = 1000;
const float ACCEL_Y = 10000;
// sheet height 
const int MAX_STEPS_Y = 10000;
int stateY;
int currentYGoal;
int nextYGoal;

// daisy wheel
const byte DIR_PIN_Z = 7;
const byte STEP_PIN_Z = 4;
const float MAX_SPEED_Z = 700;
const float HOMING_SPEED_Z = MAX_SPEED_Z;
const int NUMBER_LETTERS = 100;
//steps for one full rotation, 100 letters on wheel
const int MAX_STEPS_Z = 100;
// steps to take from startup jam position, to '.' as
// current position, '.' is home position
const float START_OFFSET_Z = 100;
const float ACCEL_Z = 10000;
int stateZ;
int currentZGoal;
int nextZGoal;

const byte LIMIT_X_AXIS_PIN = 9; 

const byte STEPPER_ENABLE_PIN = 8;

const byte HAMMER_PIN = 11;
int currentHamGoal;
int nextHamGoal;
const byte MAX_HAM_LEVEL = 5;


AccelStepper stepperA(AccelStepper::DRIVER, STEP_PIN_A, DIR_PIN_A);
AccelStepper stepperX(AccelStepper::DRIVER, STEP_PIN_X, DIR_PIN_X);
AccelStepper stepperY(AccelStepper::DRIVER, STEP_PIN_Y, DIR_PIN_Y);
AccelStepper stepperZ(AccelStepper::DRIVER, STEP_PIN_Z, DIR_PIN_Z);

const byte numChars = 32;
char receivedCommand[numChars];

boolean newCommand = false;
boolean startUp = true;

// put function declarations here:
void runStateMachines();
void startCommand();
void recvCommand();
void processNewCommand();
void confirmCommandRecieved();
bool allAreWaiting();
bool allAtPosition();
void doStartup();

void setup() {
  pinMode(STEPPER_ENABLE_PIN, OUTPUT);
  // HIGH means disabled
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

  // stepperZ.setMaxSpeed(MAX_SPEED_Z);
  // stepperZ.setAcceleration(ACCEL_Z);
  stateZ = WAITING;


  Serial.begin(9600);
  Serial.println("<Arduino is ready>");
}


// TODO remove, for testing
boolean start = true;

void loop(){
  if(start){
    digitalWrite(STEPPER_ENABLE_PIN, LOW);
    stepperZ.moveTo(10);
    stepperZ.setSpeed(10);
    start = false;
  }
  else if (stepperZ.run()){
    stepperZ.run();
  } else {
    digitalWrite(STEPPER_ENABLE_PIN, HIGH);
  
  }
}
// void loop() {
//   doStartUp();

//   if(allAreWaiting()){
//     recvCommand();
//     processNewCommand(&currentXGoal, &currentYGoal, &currentZGoal, &currentHamGoal); 
//     startCommand();
//   }
//   // else if(allAtPosition()){
//   //   recvCommand();
//   //   processNewCommand(&nextXGoal, &nextXGoal, &nextYGoal, &nextHamGoal);

//   // }

//   runStateMachines();
// }

void runStateMachines(){
  switch(stateA){ // TODO implement stop switch for ribbon sensor
    case RUNNING:
      if(stepperA.distanceToGo() == 0){
        stateA = AT_POS;
        break;
      }      
      stepperA.run();
  }
  switch(stateX){
    case RUNNING:
      if(stepperX.distanceToGo() == 0){
        stateX = AT_POS;
        break;
      }
      stepperX.run();
  }
  switch(stateY){
    case RUNNING:
      if(stepperY.distanceToGo() == 0){
        stateY = AT_POS;
        break;
      }
      stepperY.run();
  }
  switch(stateZ){
    case RUNNING:
      if(stepperZ.distanceToGo() == 0){
        stateZ = AT_POS;
        break;
      }
      stepperZ.run();
  }

  if(allAtPosition()){ //trigger hammer
    // TODO find out how to implement without delay calls
  }
}

void startCommand(){
  stepperA.move(INCR_SIZE_A);
  stateA = RUNNING;
  
  stepperX.moveTo(currentXGoal);
  stateX = RUNNING;

  stepperY.moveTo(currentYGoal);
  stateY = RUNNING;

  // TODO think about circular coordinate system for Z (daisy wheel) for faster movemnt
  stepperZ.moveTo(currentZGoal);
  stateY = RUNNING;
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

// writes goals from commands into variables
void processNewCommand(int *XGoal, int *YGoal, int *ZGoal, int *HamGoal) {
  if (newCommand == true) {
    char* commandTmp = strtok(receivedCommand, " ");
    for (int i = 0; i< 4 && commandTmp != 0; ){
      switch(commandTmp[0]){
        case 'X':
          *XGoal = atoi(&commandTmp[1]);
          break;
        case 'Y':
          *YGoal = atoi(&commandTmp[1]);
          break;
        case 'L':
          *ZGoal = max(max(0, atoi(&commandTmp[1])), NUMBER_LETTERS-1);
          break;
        case 'T':
          *HamGoal = max(max(1, atoi(&commandTmp[1])), MAX_HAM_LEVEL);
          break;
        default:
          break;
      }

      commandTmp = strtok(NULL, " ");
    }
    newCommand = false;
  }
}

void confirmCommandRecieved() {
  Serial.write("A");
  }

bool allAreWaiting() {
  return stateA == WAITING && stateX == WAITING && stateY == WAITING && stateZ == WAITING;
}
bool allAtPosition(){
  return stateA == AT_POS && stateX == AT_POS && stateY == AT_POS && stateZ == AT_POS;
}

void doStartUp(){
  if(startUp){
    // move daisy wheel and horizontal movement to home position
    switch(stateX){ //1.  horizontal movement
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
        stepperX.run();       
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
          stepperZ.move(START_OFFSET_Z);
          stateZ = RUN_HOME;
          break;
        }
        stepperZ.run();
        break;
      case RUN_HOME:
        if(stepperZ.distanceToGo() == 0){
          stepperZ.setCurrentPosition(0);
          stateZ = AT_POS;
        }
        stepperZ.run();
        break;
      default:
        break;
    }
  }
}
