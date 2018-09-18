import java.util.ArrayList;
import java.util.Random;

// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

class Main
{
	public static void main(String[] args)
	{
		SupervisedLearner timeSeries = new NeuralNet();
		timeSeries.train(null, null);
		
		Matrix labels = new Matrix();
		labels.loadARFF("labor_stats.arff");
		int iteration =350;
                double learning_rate = 0.01;
                double lembda = 0.001;
		for (int p = 0; p < iteration; p++) {
			for (int i = 0; i < 256; i++) {
				Vec x = new Vec(1);
				x.set(0, i / 256.0);
				Vec y = new Vec(labels.row(i).vals);
				((NeuralNet) timeSeries).L2Regularization(x, y, learning_rate, lembda);
			}
		}
		
		for(int i = 0; i < 356; i++)
		{
			Vec x = new Vec(1);
			x.set(0, i/256.0);
                        
			Vec y = new Vec(labels.row(i).vals);
			Vec pred = ((NeuralNet)timeSeries).predict(x);
			System.out.println(pred.toString());
		}
		
	}
}
