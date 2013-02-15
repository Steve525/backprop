package Backpropagation;

public class HiddenUnit extends AbstractUnit {

	private double _errorSum;
	
	public HiddenUnit(int number) {
		_id = number;
		_errorSum = 0;
	}

	
	public void receiveError(double weight_kh, double errorTerm_k) {
		_errorSum += (weight_kh * errorTerm_k);
	}


	@Override
	public void calculateErrorTerm() {
		_errorTerm = _output * (1 - _output) * _errorSum;
		_errorSum = 0;
	}
}
