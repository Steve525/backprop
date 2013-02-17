package Backpropagation;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import Main.Matrix;
import Main.SupervisedLearner;


public class NeuralNet extends SupervisedLearner {

	private NetworkManager _networkManager;
	private double _learningRate;
	private double _momentum;
	private double _maxWeightRange;
	private double _minWeightRange;
	
	public NeuralNet(Random rand) {
		_learningRate = 0.3;
		_momentum = 0;
		_maxWeightRange = 0.05;
		_minWeightRange = -0.05;
		_networkManager = new NetworkManager(_learningRate, _momentum, _maxWeightRange, _minWeightRange);
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		runBackPropagationTraining(features, labels);
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = _networkManager.getNetworkOutput(features);
	}
	
	private void runBackPropagationTraining (Matrix inputs, Matrix targets) throws IOException {
		// initialize the neural network to have as many input nodes as features and
		//	as many output units as there are output classes
		_networkManager.initializeNetwork(inputs.cols(), 2*inputs.cols(), 1, targets.valueCount(0),
											inputs.rows());
		String sFileName = "LearningRateTest-" + Double.toString(_learningRate) + ".csv";
		FileWriter writer = new FileWriter(sFileName);
		double mse = 0;
		double prev_mse = 0;
		double classificationAccuracy = 0;
		int epochs = 0;
		boolean stoppingCriteriaMet = false;
		//while (!stoppingCriteriaMet) {
		for (int i = 0; i < 2000; i++) {
			prev_mse = _networkManager.getPrevNetworkMSE();
			runAnEpoch(inputs, targets);
			epochs++;
			mse = _networkManager.getNetworkMSE();
			classificationAccuracy = _networkManager.getClassificationAccuracy();
			writer.append(Integer.toString(epochs));
			writer.append(",");
			writer.append(Double.toString(1 - classificationAccuracy));
			writer.append(",");
			writer.append(Double.toString(mse));
			writer.append("\n");
			writer.flush();
			/*System.out.print(epochs);
			System.out.print(",");
			System.out.print(1-classificationAccuracy);
			System.out.print(",");
			System.out.print(mse);
			System.out.println();*/
			//stoppingCriteriaMet = stoppingCriteriaMet(prev_mse, mse);
		}
		writer.close();
	}
	
	public void runAnEpoch (Matrix inputs, Matrix targets) {
		for (int i = 0; i < inputs.rows(); i++) {
			_networkManager.propagateInputForward(inputs.row(i), targets.row(i));
			_networkManager.keepTrackOfOutputUnitErrorSum();
			//_networkManager.printInputOutput(inputs.row(i), targets.row(i));
			//System.out.println("Forward propagating...");
			//_networkManager.printPredictedOutput();
			//System.out.println("Back propagating...");
			_networkManager.propagateErrorsBack();
			//_networkManager.printErrorTerms();
			//System.out.println("Descending gradient...");
			_networkManager.updateAllWeights();
		}
	}
	
	public boolean stoppingCriteriaMet(double prevError, double error) {
		return Math.abs(prevError - error) < 0.01;
	}

}
