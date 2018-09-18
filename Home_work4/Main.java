import java.util.Random;

class Main  {

    public static void main(String[] args) {
	
	SupervisedLearner ConvolutionNeuralNet = new neuralNet_convolutionNetwork();
	ConvolutionNeuralNet.train(null, null);
	int inputVectorLength= 8*8;
        int outputVectorLength= 3;
			
	Vec inputVector = new Vec(inputVectorLength);
        //    System.out.println("............."+inputVector.len);
	Vec outputVector = new Vec(outputVectorLength);
	Random random = new Random();	
	for(int i = 0; i < inputVectorLength; i++){
            inputVector.set(i, random.nextGaussian());
            //        System.out.println("Main.main()");
        }
	for(int i = 0; i < outputVectorLength; i++){
            outputVector.set(i, random.nextGaussian());
            //        System.out.println("Main.main()");
        }
	((neuralNet_convolutionNetwork) ConvolutionNeuralNet).refineWeights(inputVector, outputVector);	
    }
}



