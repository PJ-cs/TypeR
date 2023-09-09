#include <Arduino.h>
#include <AccelStepper.h>


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
#define INCR_SIZE_A -5
// pin for ribbon sensor, triggers if ribbon is empty
#define LIMIT_A_AXIS_PIN -1

// carriage, horizontal movement
#define DIR_PIN_X 5
#define STEP_PIN_X 2
#define MAX_SPEED_X 1000.0
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
#define MAX_SPEED_Y 1000.0
#define ACCEL_Y 10000
// sheet height 
#define MAX_STEPS_Y 10000
#define STEPS_PER_PIXEL_Y 17

// daisy wheel
#define Z_PIN_4 12 //SPIN enable
#define Z_PIN_3 7 // dirz
#define Z_PIN_2 4 // stepz
#define Z_PIN_1 10 //y+
#define MAX_SPEED_Z 100
#define HOMING_SPEED_Z 700
#define NUMBER_LETTERS 100
#define STEP_SIZE_Z  1
// steps to take from startup jam position '*', to '.' as
// current position, '.' is home position
#define START_OFFSET_Z (50 * STEP_SIZE_Z)
//steps for one full rotation, 100 letters on wheel
#define MAX_STEPS_Z (NUMBER_LETTERS * STEP_SIZE_Z)
#define ACCEL_Z 1500

#define STEPPER_ENABLE_PIN 8

#define HAMMER_PIN 11
#define NUM_O_HAM_LEVEL 5
//minimal time to wait until next hammer hit
#define HAM_COOL_MS 30
//minimal activiation time of hammer pin to hit paper
#define HAM_ACT_MS 15

int stateA;
AccelStepper stepperA(AccelStepper::DRIVER, STEP_PIN_A, DIR_PIN_A);

int stateX;
int currentXGoal;
int nextXGoal;
AccelStepper stepperX(AccelStepper::DRIVER, STEP_PIN_X, DIR_PIN_X);

int stateY;
int currentYGoal;
int nextYGoal;
AccelStepper stepperY(AccelStepper::DRIVER, STEP_PIN_Y, DIR_PIN_Y);

int stateZ;
int currentZGoal;
int nextZGoal;
AccelStepper stepperZ(AccelStepper::FULL4WIRE, Z_PIN_1, Z_PIN_2, Z_PIN_3, Z_PIN_4);

int stateHam;
unsigned long startMillis;
// contains the tupel (#hits, strength)
uint8_t currentHamGoal[2];
uint8_t nextHamGoal[2];
// five pairs of (#hits, strength[200-250])
const uint8_t hamLevels[5][2] = {{1, 200}, {1, 230}, {1, 250}, {2, 250}, {3, 250}};


const uint8_t numChars = 32;
char receivedCommand[numChars];

boolean newCommandToRead = false;
boolean newGoalsSet = false;

boolean startUpRunning = false;

// put function declarations here:
void runStateMachines();
void startCommand();
void recvCommand();
void processNewCommand(int *XGoal, int *YGoal, int *ZGoal, uint8_t *HamGoal);
void confirmCommandRecieved();
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
        processNewCommand(&currentXGoal, &currentYGoal, &currentZGoal, currentHamGoal); 
        startCommand();
      }
      // else if(allAtPosition()){
      //   recvCommand();
      //   processNewCommand(&nextXGoal, &nextXGoal, &nextYGoal, &nextHamGoal);

      // }

      runStateMachines();
  }
  
}

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
      stepperZ.runSpeedToPosition();
  }

  if(allStepperAtPosition()){ //trigger hammer
    //switch of stepper motor for the duration of hammer
    digitalWrite(STEPPER_ENABLE_PIN, HIGH);
    switch(stateHam){
      case WAITING:
        if(currentHamGoal[0] > 0){
          stateHam = RUNNING;
          analogWrite(HAMMER_PIN, currentHamGoal[1]);
          startMillis = millis();
          currentHamGoal[0]--;
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
        }
        break;
    }
  }
}

void startCommand(){
  if(newGoalsSet){
    digitalWrite(STEPPER_ENABLE_PIN, LOW);
    stepperA.move(INCR_SIZE_A);
    stateA = RUNNING;
    
    stepperX.moveTo(currentXGoal);
    stateX = RUNNING;

    stepperY.moveTo(currentYGoal);
    stateY = RUNNING;

    // circular coordinate system for Z (daisy wheel) for faster movemnt
    // if(abs(currentZGoal - stepperZ.currentPosition()) > MAX_STEPS_Z / 2){
    //   if(currentZGoal > stepperZ.currentPosition()){
    //     stepperZ.setCurrentPosition(stepperZ.currentPosition()+ MAX_STEPS_Z);
    //   }
    //   else{
    //     stepperZ.setCurrentPosition(stepperZ.currentPosition()- MAX_STEPS_Z);
    //   }
    // }
   
    stepperZ.moveTo(currentZGoal);
     if(abs(currentZGoal- stepperZ.currentPosition()) < MAX_STEPS_Z){
      stepperZ.setSpeed(MAX_SPEED_Z/4);
    }else{
      stepperZ.setSpeed(MAX_STEPS_Z-1);
    }
    stateZ = RUNNING;

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
void processNewCommand(int *XGoal, int *YGoal, int *ZGoal, uint8_t *hamGoal) {
  if (newCommandToRead == true) {
    char* commandTmp = strtok(receivedCommand, " ");
    for (int i = 0; i< 4 && commandTmp != 0; ){
      switch(commandTmp[0]){
        case 'X':{
          *XGoal = min(max(0, (atoi(&commandTmp[1]) * STEPS_PER_PIXEL_X)), MAX_STEPS_X);
          
        }break;
        case 'Y':{
          *YGoal = -min(max(0, (atoi(&commandTmp[1]) * STEPS_PER_PIXEL_Y)), MAX_STEPS_Y);
        }break;
        case 'L': {
          *ZGoal = min(max(0, atoi(&commandTmp[1])), NUMBER_LETTERS-1) * STEP_SIZE_Z;
        }break;
        case 'T':{
          uint8_t hamLevel = min(max(0, atoi(&commandTmp[1]))-1, NUM_O_HAM_LEVEL-1);
          hamGoal[0] = hamLevels[hamLevel][0];
          hamGoal[1] = hamLevels[hamLevel][1];
        }break;
      }
      commandTmp = strtok(NULL, " ");
    }
    newCommandToRead = false;
    newGoalsSet = true;
  }
}

void confirmCommandRecieved() {
  Serial.print("A");
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
          stepperZ.moveTo(MAX_STEPS_Z*4); 
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
          stepperZ.setCurrentPosition(0);
          startUp = false;
          digitalWrite(STEPPER_ENABLE_PIN, HIGH);
          stateZ = WAITING;
          stateX = WAITING;
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