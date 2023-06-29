class Controller
{
private:
    void doStartUP();

public:
    Controller();
    ~Controller();
    bool executeCommand(int XGoal, int YGoal, int ZGoal, int HamGoal);
};

Controller::Controller(/* args */)
{
    doStartUP();
}

Controller::~Controller()
{
}

void Controller::doStartUP()
{
}

bool executeCommand(int XGoal, int YGoal, int ZGoal, int HamGoal){}