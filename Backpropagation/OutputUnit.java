package Backpropagation;

public class OutputUnit extends AbstractUnit {

	double _target;
	
	public OutputUnit(int number) {
		_id = number;
		_target = 0;
	}

	@Override
	public void calculateErrorTerm() {
		_errorTerm = _output * (1 - _output) * (_target - _output);
	}
	
	public void setTarget(double target) {
		_target = target;
	}
	
	public double getTarget() { return _target; }
}
