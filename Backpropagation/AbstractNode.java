package Backpropagation;

public abstract class AbstractNode implements INode {

	double _output;
	
	public AbstractNode() {
		_output = 0;
	}
	
	public double getOutput() {
		return _output;
	}
}
