#include <AccelStepper.h>

class CustomStepper : public AccelStepper {
public:
    CustomStepper(uint8_t interface = AccelStepper::FULL4WIRE, uint8_t pin1 = 2, uint8_t pin2 = 3, uint8_t pin3 = 4, uint8_t pin4 = 5, bool enable = true):
        AccelStepper(interface,pin1, pin2, pin3, pin4, enable){
    }

    
    void step4(long step) override
    {
        switch (step & 0x3)
        {
        case 2:    // 1010u
            setOutputPins(0b0110);
            break;

        case 3:    // 0110
            setOutputPins(0b1010);
            break;

        case 0:    //0101
            setOutputPins(0b1001);
            break;

        case 1:    //1001
            setOutputPins(0b0101);
            break;
        }
    }
};