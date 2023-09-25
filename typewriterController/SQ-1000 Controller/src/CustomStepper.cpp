#include <AccelStepper.h>

class CustomStepper : public AccelStepper {
    void step4(long step)
{
    switch (step & 0x3)
    {
	case 0:    // 1010
	    setOutputPins(0b1010);
	    break;

	case 1:    // 0110
	    setOutputPins(0b0110);
	    break;

	case 2:    //0101
	    setOutputPins(0b0101);
	    break;

	case 3:    //1001
	    setOutputPins(0b1001);
	    break;
    }
}
};