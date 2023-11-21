#include <Arduino.h>
#include <AccelStepper.h>
#include "CustomStepper.h"

// Stepper State definitions
#define RUNNING 1
#define WAITING 2
#define STOPPED 3
#define AT_POS 4
#define RUN_HOME 5
#define AT_HOME 6
#define COOL_DOWN 7
#define AT_ENDSTOP 8
#define AT_ENDSTOP1 9
#define AT_ENDSTOP2 10
#define RIBBON_ERROR 11


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
#define DIR_PIN_A 13 
#define STEP_PIN_A 12
#define MAX_SPEED_A 1000.0
#define ACCEL_A 10000
//distance to move to have fresh ink ribbon
//does not have goals, moves always same amount of steps
#define INCR_SIZE_A (-5*5)
// pin for ribbon sensor, triggers if ribbon is empty
#define LIMIT_A_AXIS_PIN A1

// carriage, horizontal movement
#define DIR_PIN_X 5
#define STEP_PIN_X 2
#define MAX_SPEED_X 1500.0
#define HOMING_STEPS_X -100000
// how many steps to the right of stop
// switch to take, where new home pos
// if decrease OFFSET -> increase MAX steps by same amount
#define START_OFFSET_X 300
#define ACCEL_X 20000
#define MAX_STEPS_X 17700
// sheet width
#define STEPS_PER_PIXEL_X 10
#define LIMIT_X_AXIS_PIN 9 

// line feed, vertical movement
#define DIR_PIN_Y 6
#define STEP_PIN_Y 3
#define MAX_SPEED_Y 3000.0
#define ACCEL_Y 20000
// sheet height 
#define MAX_STEPS_Y 30000 // TODO how to handle, allow no limit?
#define STEPS_PER_PIXEL_Y 17

// daisy wheel
#define Z_PIN_4 A3 //Cool Enable
#define Z_PIN_3 7 // dirz
#define Z_PIN_2 4 // stepz
#define Z_PIN_1 10 //y+
#define MAX_SPEED_Z 300
#define HOMING_SPEED_Z 700
#define NUMBER_LETTERS 100
#define STEP_SIZE_Z  1
// steps to take from startup jam position '*', to '.' as
// current position, '.' is home position
#define START_OFFSET_Z (51 * STEP_SIZE_Z)
//steps for one full rotation, 100 letters on wheel
#define MAX_STEPS_Z (NUMBER_LETTERS * STEP_SIZE_Z)
#define ACCEL_Z 1500

#define STEPPER_ENABLE_PIN 8

#define HAMMER_PIN 11
#define NUM_O_HAM_LEVEL 3
#define HAM_FACTOR 1.0
#define MAX_HAM_STR 230   // 0- 255, limit strength of hammer this is ca. 10.4 V
#define MIN_HAM_STR 160 // for some letters, six instead of 9, at least

//minimal time to wait until next hammer hit
#define HAM_COOL_MS 30
//minimal activiation time of hammer pin to hit paper
#define HAM_ACT_MS 15

int stateA;
AccelStepper stepperA(AccelStepper::DRIVER, STEP_PIN_A, DIR_PIN_A);

int stateX;
int currentXGoal;
AccelStepper stepperX(AccelStepper::DRIVER, STEP_PIN_X, DIR_PIN_X);

int stateY;
int currentYGoal;
AccelStepper stepperY(AccelStepper::DRIVER, STEP_PIN_Y, DIR_PIN_Y);

int stateZ;
uint8_t currentZGoal;
CustomStepper stepperZ(AccelStepper::FULL4WIRE, Z_PIN_1, Z_PIN_2, Z_PIN_3, Z_PIN_4);

int stateHam;
unsigned long startMillis;
// contains the tupel (# full strength hits, strength of last partial hit)
uint8_t currentHamGoal;
uint8_t activeHamGoal;
// order: ".", ">", "‰", "<", "|", "'", "³", "_", "Y" ..., order 0,1,2,3,4 ...
const float area_letters[100] = {1.61, 4.41, 8.74, 4.4, 5.24, 1.47, 3.58, 3.2, 6.46, 5.91, 
                                  4.79, 7.82, 8.86, 6.53, 8.95, 8.85, 7.44, 9.08, 8.59, 7.77, 
                                  8.07, 5.52, 7.54, 8.54, 7.29, 6.45, 8.5, 7.17, 6.6, 8.73, 
                                  7.87, 7.64, 7.17, 9.36, 7.14, 8.2, 9.4, 8.61, 3.66, 3.71, 
                                  2.76, 8.73, 7.34, 4.33, 6.28, 6.63, 3.52, 4.21, 4.21, 1.0, 
                                  3.69, 3.45, 3.23, 7.38, 3.54, 2.6, 7.1, 5.63, 7.88, 6.6, 
                                  7.73, 5.58, 8.21, 7.18, 8.64, 8.68, 7.21, 6.79, 6.91, 4.9, 
                                  6.41, 7.23, 6.08, 5.01, 5.29, 5.47, 6.02, 6.43, 8.11, 7.93, 
                                  6.55, 8.55, 5.98, 7.16, 8.23, 7.41, 2.29, 4.71, 6.4, 5.91, 
                                  6.49, 6.88, 6.53, 7.13, 4.46, 7.9, 7.14, 1.0, 4.06, 6.6};

const uint8_t numChars = 32;
char receivedCommand[numChars];

boolean newCommandToRead = false;
boolean newGoalsSet = false;

boolean startUpRunning = false;

// put function declarations here:
void runStateMachines();
void startCommand();
void recvCommand();
void processNewCommand(int *XGoal, int *YGoal, uint8_t *ZGoal, uint8_t *HamGoal);
void confirmCommandRecieved();
void sentRibbonError();
bool allAreWaiting();
bool allStepperAtPosition();
boolean doStartUpOnce();

void setup() {
  pinMode(STEPPER_ENABLE_PIN, OUTPUT);

  pinMode(HAMMER_PIN, OUTPUT);
  analogWrite(HAMMER_PIN, 0);

  pinMode(LIMIT_X_AXIS_PIN, INPUT_PULLUP);
  pinMode(LIMIT_A_AXIS_PIN, INPUT);

  
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
  stepperZ.setMaxSpeed(MAX_SPEED_Z);
  //stepperZ.setSpeed(MAX_SPEED_Z);
  stateZ = WAITING;

  stateHam = WAITING;

  Serial.begin(9600);
  // HIGH means disabled for the enable pin
  digitalWrite(STEPPER_ENABLE_PIN, LOW);
  delay(500);
  Serial.println("<Arduino is ready>");
}


void loop() {
  startUpRunning = doStartUpOnce();

  if (!startUpRunning){
    if(allAreWaiting()){
      recvCommand();
      processNewCommand(&currentXGoal, &currentYGoal, &currentZGoal, &currentHamGoal); 
      startCommand();
    } 
    runStateMachines();
    
      // else if(allAtPosition()){
      //   recvCommand();
      //   processNewCommand(&nextXGoal, &nextXGoal, &nextYGoal, &nextHamGoal);

      // }

  }
  
}

void runStateMachines(){
  static int a_limit_not_triggered = 0;
  switch(stateA){ // TODO implement stop switch for ribbon sensor
    case RUNNING:
      if(!digitalRead(LIMIT_A_AXIS_PIN)){
        stateA = AT_ENDSTOP;
        a_limit_not_triggered = 0;
        break;
      }
      if(stepperA.distanceToGo() == 0){
        a_limit_not_triggered++;
        if (a_limit_not_triggered > 6){
          stateA = RIBBON_ERROR;
          a_limit_not_triggered = 0;
          sentRibbonError();
          break;
        }else{
          stateA = AT_POS;
          break;
        }
      }      
      stepperA.run();
      break;
    case AT_ENDSTOP:
      if(stepperA.distanceToGo() == 0){
        stateA = AT_POS;
        break;
      }      
      stepperA.run();
      break;
    case RIBBON_ERROR:
      if(Serial.read() == 'C'){
        stateA = AT_POS;
      }
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
      stepperZ.runSpeedToPosition();
  }

  if(allStepperAtPosition()){ //trigger hammer
    //switch of stepper motor for the duration of hammer
    delay(50); // TODO: shorter
    digitalWrite(STEPPER_ENABLE_PIN, HIGH);
    stepperZ.disableOutputs();
    switch(stateHam){
      case WAITING:
        if (activeHamGoal > 0){
          stateHam = RUNNING;
          analogWrite(HAMMER_PIN, MIN_HAM_STR + activeHamGoal);
          startMillis = millis();
          // Serial.print(currentZGoal);
          // Serial.println(" letter hammered.");
        }
        else{
          stateHam = WAITING;
          stateA = WAITING;
          stateX = WAITING;
          stateY = WAITING;
          stateZ = WAITING;
          confirmCommandRecieved(); // TODO find more efficient method
        }
        break;
      case RUNNING:
        if(millis() - startMillis >= HAM_ACT_MS){
          analogWrite(HAMMER_PIN, 0);
          stateHam = COOL_DOWN;
          startMillis = millis();
        }
        break;
      case COOL_DOWN:
        if(millis() - startMillis >= HAM_COOL_MS){
          stateHam = WAITING;
          stateA = WAITING;
          stateX = WAITING;
          stateY = WAITING;
          stateZ = WAITING;
          confirmCommandRecieved(); // TODO find more efficient method
        }
        break;
    }
  }
}

void startCommand(){
  if(newGoalsSet){
    digitalWrite(STEPPER_ENABLE_PIN, LOW);
    stepperZ.enableOutputs();
    stepperA.move(INCR_SIZE_A);
    stateA = RUNNING;
    
    stepperX.moveTo(currentXGoal);
    stateX = RUNNING;

    stepperY.moveTo(currentYGoal);
    stateY = RUNNING;

    // circular coordinate system for Z (daisy wheel) for faster movemnt
    if(abs(currentZGoal - stepperZ.currentPosition()) > MAX_STEPS_Z / 2){
      if(currentZGoal > stepperZ.currentPosition()){
        stepperZ.setCurrentPosition(stepperZ.currentPosition()+ MAX_STEPS_Z+2);
      }
      else{
        stepperZ.setCurrentPosition(stepperZ.currentPosition()- MAX_STEPS_Z-2);
      }
    }
   
    stepperZ.moveTo(currentZGoal);
    stepperZ.setSpeed(MAX_STEPS_Z);
    
    stateZ = RUNNING;

    activeHamGoal = currentHamGoal;

    newGoalsSet = false;
  }
}

void recvCommand() {
  static boolean recvInProgress = false;
  static byte ndx = 0;
  char startMarker = '<';
  char endMarker = '>';
  char rc;

  while (Serial.available() > 0 && newCommandToRead == false) {
      rc = Serial.read();

      if (recvInProgress == true && rc != startMarker) {
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
              newCommandToRead = true;
          }
      }

      else if (rc == startMarker) {
          recvInProgress = true;
      }
  }
}

// writes goals from commands into variables
void processNewCommand(int *XGoal, int *YGoal, uint8_t *ZGoal, uint8_t *hamGoal) {
  if (newCommandToRead == true) {
    char* commandTmp = strtok(receivedCommand, " ");
    for (int i = 0; i< 4 && commandTmp != 0; ){
      switch(commandTmp[0]){
        case 'X':
          *XGoal = min(max(0, (atoi(&commandTmp[1]) * STEPS_PER_PIXEL_X)), MAX_STEPS_X);
          break;
        
        case 'Y':
          *YGoal = -min(max(0, (atoi(&commandTmp[1]) * STEPS_PER_PIXEL_Y)), MAX_STEPS_Y);
          break;
        
        case 'L': 
          *ZGoal = min(max(0, atoi(&commandTmp[1])), NUMBER_LETTERS-1) * STEP_SIZE_Z;
          break;
        
        case 'T':
          int hamLevel = min(max(0, atoi(&commandTmp[1])),255);
          float hamLevelNorm = hamLevel / 255.;
          // float letterArea = area_letters[*ZGoal];
          uint8_t hammerPts = (MAX_HAM_STR-MIN_HAM_STR) * hamLevelNorm ;
          float areaFac =  area_letters[*ZGoal] / 4. ;
          hammerPts =  areaFac > 1. ?  hammerPts * areaFac : hammerPts;
          if((*ZGoal != 97 ||
              *ZGoal != 86 ||
              *ZGoal != 49 ||
              *ZGoal != 5 ||
              *ZGoal != 0) && hammerPts > 0){
            hammerPts += 6;
          }          
          *hamGoal = hammerPts;// % (MAX_HAM_STR-MIN_HAM_STR);
          break;
      }
      commandTmp = strtok(NULL, " ");
    }
    newCommandToRead = false;
    newGoalsSet = true;
  }
}

void confirmCommandRecieved() {
  Serial.print("A: ");
  Serial.print(currentXGoal);
  Serial.print(" ");
  Serial.print(currentYGoal);
  Serial.print(" ");
  Serial.print(currentZGoal);
  Serial.print(" ");
  Serial.println(currentHamGoal+MIN_HAM_STR);
}

void sentRibbonError() {
  Serial.println("R");
}

bool allAreWaiting() {
  return stateA == WAITING && stateX == WAITING && stateY == WAITING && stateZ == WAITING &&
  stateHam == WAITING;
}
bool allStepperAtPosition(){
  return stateA == AT_POS && stateX == AT_POS && stateY == AT_POS && stateZ == AT_POS;
}

boolean doStartUpOnce(){
  static boolean startUp = true;
  if(startUp){
    // move daisy wheel and horizontal movement to home position
    switch(stateX){ //horizontal movement
      case WAITING:
        //Serial.write("X: WAITING\n");
        stepperX.move(HOMING_STEPS_X);
        stateX = RUNNING;
        break;
      case RUNNING:
        //Serial.write("X: RUNNING\n");
        if(digitalRead(LIMIT_X_AXIS_PIN) == LOW){
          //Serial.println("X limit switch triggerd");
          stepperX.stop();
          stepperX.runToPosition();
          stateX = AT_ENDSTOP;
          break;
        }
        stepperX.run();
        break;
      case AT_ENDSTOP:
        //Serial.write("X: AT_ENDSTOP\n");
        if(stateZ == AT_ENDSTOP){
          stepperX.move(START_OFFSET_X);
          stateX = RUN_HOME;
        }
        break;
      case RUN_HOME:
        //Serial.write("X: RUN_HOME\n");
        if(stepperX.distanceToGo() == 0){
          stateX = AT_HOME;
          stepperX.setCurrentPosition(0);
          break;
        } 
        stepperX.run();       
        break;
    }

    switch(stateZ){
      case WAITING:
        //Serial.write("Z: WAITING\n");
        if(stateX == AT_ENDSTOP){
          //stepperZ.setCurrentPosition(1);
          // if horizontal at stop switch, try rotate daisy wheel one time,
          // will be physically blocked at '*' pos, then reset
          stepperZ.moveTo(-MAX_STEPS_Z*2); 
          stepperZ.setSpeed(MAX_SPEED_Z);
          stateZ = RUNNING;
        }
        break;
      case RUNNING:
        //Serial.println("Z: RUNNING\n");
        if(stepperZ.distanceToGo() == 0){
          stepperZ.setCurrentPosition(0);
          stateZ = AT_ENDSTOP;
          delay(2000);
          break;
        }
        stepperZ.runSpeedToPosition();
        break;

      case AT_ENDSTOP:
        //Serial.write("Z: AT_ENDSTOP\n");
        if(stateX == AT_HOME){
          stepperZ.moveTo(START_OFFSET_Z);
          stepperZ.setSpeed(MAX_SPEED_Z);
          stateZ = RUN_HOME;
        }
        break;
      case RUN_HOME:
        //Serial.write("Z: RUN_HOME\n");
        if(stepperZ.distanceToGo() == 0){
          startUp = false;
          digitalWrite(STEPPER_ENABLE_PIN, HIGH);
          stepperZ.disableOutputs();
          stateZ = WAITING;
          stateX = WAITING;
          delay(1000);
          stepperZ.setCurrentPosition(-2);
          break;
        }
        stepperZ.runSpeedToPosition();
        break;
      default:
        break;
    }
  }
  return startUp;
}


// #include <AccelStepper.h>
// #define DIR_PIN_Z 7
// #define STEP_PIN_Z 4
// #define STEPPER_ENABLE_PIN 8

// AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN_Z, DIR_PIN_Z);
// void setup()
// {  
//   stepper.setMaxSpeed(180);
//   stepper.setAcceleration(800);
//   pinMode(STEPPER_ENABLE_PIN, OUTPUT);
//   digitalWrite(STEPPER_ENABLE_PIN, LOW);
// }
 
// void loop()
// { 
//   stepper.setCurrentPosition(1);   
//   stepper.moveTo(50);
//   while (stepper.currentPosition() != 50) // Full speed up to 300
//     stepper.run();
//   stepper.stop(); // Stop as fast as possible: sets new target
//   stepper.runToPosition(); 
//   delay(10000);
//   // Now stopped after quickstop
 
//   // Now go backwards
//   stepper.moveTo(1);
//   while (stepper.currentPosition() != 1) // Full speed basck to 0
//     stepper.run();
//   stepper.stop(); // Stop as fast as possible: sets new target
//   stepper.runToPosition(); 
//   delay(10000);
//   // Now stopped after quickstop
//   // stepper.runToNewPosition(50);
   
//   // wait();
//   // stepper.runToNewPosition(0);
//   // wait();
 
// }

// void wait(){
//    digitalWrite(STEPPER_ENABLE_PIN, HIGH);

//   delay(10000);
//     digitalWrite(STEPPER_ENABLE_PIN, LOW);
// }


// //Includes the Arduino Stepper Library
// #include <Stepper.h>

// // Defines the number of steps per rotation
// const int stepsPerRevolution = 2038;

// // Creates an instance of stepper class
// // Pins entered in sequence IN1-IN3-IN2-IN4 for proper step sequence
// Stepper myStepper = Stepper(stepsPerRevolution, 8, 10, 9, 11);

// void setup() {
//     // Nothing to do (Stepper Library sets pins as outputs)
// }

// void loop() {
// 	// Rotate CW slowly at 5 RPM
// 	myStepper.setSpeed(5);
// 	myStepper.step(stepsPerRevolution);	
//   	delay(1000);

// 	// Rotate CCW quickly at 10 RPM
// 	myStepper.setSpeed(10);
// 	myStepper.step(-stepsPerRevolution);
// 	delay(1000);
// }