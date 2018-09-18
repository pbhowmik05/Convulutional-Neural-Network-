import java.util.ArrayList;
import java.util.Random;

public class NeuralNet extends SupervisedLearner {

	ArrayList<Layer> mLayers;
	Vec weights;
	int layerWeightSize;

	@Override
	String name() {
		return "NeuralNet";
	}

        @Override
        void train(Matrix features, Matrix labels) {

            layerWeightSize = 0;
            int m_size = 0;
            int b_size = 0;
            mLayers = new ArrayList<>();
            LayerLinear L1 = new LayerLinear(1, 101);
            LayerSine L2 = new LayerSine(101,101);
            LayerLinear L3 = new LayerLinear(101, 1);

            mLayers.add(L1);
            mLayers.add(L2);
            mLayers.add(L3);

            System.out.println("mylayer size  "+mLayers.size());
            for (int i = 0; i < mLayers.size(); i=i+1) {     /// may need to revisit
            
                if(i==0||i==2) {
                    m_size = mLayers.get(i).mInput_size * mLayers.get(i).mOutput_size;
                    b_size = mLayers.get(i).mOutput_size;
                }
                else {
                    m_size = 0;
                    b_size = 0;
                }    
            //System.out.println("layerWeightSize" + (m_size + b_size));   
            layerWeightSize += (m_size + b_size);                
            }
            //System.out.println("layerWeightSize" + layerWeightSize);
            weights = new Vec(layerWeightSize);
            initWeights();
            //System.out.println("weight   :: \n" +weights.toString());
        }
     
     
    @Override
    Vec predict(Vec in) {
        
        int index = 0;
        Vec temp = new Vec(in.len);
        for(int h=0;h<in.len;h++){
            temp.set(h, in.get(h));
        }
        for (int i = 0; i < mLayers.size(); i++) {
            Vec out_layer = new Vec(0);
            int nWeight_len= 0;
            if ((i==0)||(i==2)) 
                 nWeight_len = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);
            else
                nWeight_len = 0;
                out_layer = new Vec(weights, index, nWeight_len);
           //     System.out.println("nWeight_len   : " + out_layer.toString());
                index = index + nWeight_len;
            //System.out.println("neural net x length   :" +temp.toString());
            mLayers.get(i).activate(out_layer, temp);
            temp = new Vec(mLayers.get(i).activation, 0, mLayers.get(i).activation.len);
         //  System.out.println("NeuralNet.predict()    :"  +temp.toString());
        }
        return temp;
    } 

     
   void backprop(Vec weights, Vec target) {
        
        Vec layer_blame = new Vec(target, 0, target.len);
        Vec y_actual = mLayers.get(mLayers.size() - 1).activation;  
        y_actual.scale(-1.0);
        layer_blame.add(y_actual);

        int back_index = weights.len;
        for (int i = mLayers.size() - 1; i >= 0; i--) {          
            if(i>0) {
                mLayers.get(i).blame.copy(layer_blame);
                Vec iweight = new Vec(0);
                int mod = i % 2;
                if (mod == 0) {
                    int nWeight_len = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);
                    back_index = back_index - nWeight_len;
                    iweight = new Vec(weights, back_index, nWeight_len);
                }
                //mLayers.get(i).blame.copy(layer_blame);
                layer_blame = new Vec(mLayers.get(i - 1).mOutput_size);
                mLayers.get(i).backprop(iweight, layer_blame);
            }
            else
                mLayers.get(i).blame.copy(layer_blame);
        }
    } 
      


    void updateGradient(Vec x, Vec gradient) {
        int grad_index = 0;
        //System.out.println("gradient   "+ gradient.toString());
        for (int i = 0; i < mLayers.size(); i++) {
            int grad_size = 0;
            int mod = i % 2;
         // System.out.println("mod value :  " +mod);
            if (mod == 0)
                grad_size = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);
           // System.out.println("gradient   "+ grad_size); 
            else
                grad_size = 0;

            Vec Grad_layer = new Vec(grad_size);
           // System.out.println("gradient   "+ weights.toString());           
            Grad_layer = new Vec(weights, grad_index, grad_size);
            //System.out.println("gradient   "+ Grad_layer.toString());
            if (i > 0) mLayers.get(i).updateGradient(mLayers.get(i - 1).activation, Grad_layer); 
            else  mLayers.get(i).updateGradient(x, Grad_layer);
                       
            for (int k = 0; k < grad_size; k++) {
                gradient.set(grad_index + k, Grad_layer.get(k));
            }
      //      System.out.println("NeuralNet.updateGradient()   "+ gradient.toString());
            //if (mod == 0) {  
                grad_index += grad_size;
            //}
            //System.out.println("updateGradient done\n");
        }
        
    } 

	void initWeights() {
            
            int pointer = 0;
            Random random = new Random();
            for (int i = 0; i < mLayers.size(); i++) {
                int  layer_size;
                Layer layer = mLayers.get(i);
                if( i ==0||i==2)  
                    layer_size = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);
                else
                    layer_size=0;
                int weightLen = layer_size;
                int j;
		if (i == 0) {
                    System.out.println("Layer layerLayer layerLayer layer   "+weightLen);   
                    for(j=0; j<weightLen; j++) {	
                        if(j<50)            
                            weights.set(j, Math.PI);	
                        if(j>=50 && j<100)  
                           weights.set(j, Math.PI / 2);
                        if(j==100)
                            weights.set(j, 0.0);
                        if(j>=101 && j < 151) 
                            weights.set(j, (j-100) * 2 * Math.PI);
                        if(j>=151 && j<200)
                            weights.set(j, (j-150) * 2 * Math.PI);
                        if(j==201)
                            weights.set(j, 0.01);
                    }
		} 
                else {
                    for (int m = 0; m < weightLen; m++) 
                                  //  System.out.println("weight length  "+ weightLen);
                        weights.set(pointer + m, Math.max(0.03, 1.0 / layer.mInput_size) * random.nextGaussian());				   //    System.out.println("NeuralNet.initWeights()" +weightLen);
                }        
                pointer += weightLen;
                  //      System.out.println("NeuralNet.initWeights()"+nCount );
	}
        //System.out.println("weight value" +weights.toString());
    }

	void refineWeights(Vec x, Vec y, double learningRate) {
		predict(x);
		backprop(weights, y);
		Vec gradients = new Vec(weights.len);
		gradients.fill(0);
		updateGradient(x, gradients);
		weights.addScaled(learningRate, gradients);
	}

	void L2Regularization(Vec x, Vec y, double learning_rate, double lambda) {
		predict(x);
               // Vec temp = new Vec(weights.len);
		backprop(weights, y);
		Vec gradients = new Vec(weights.len);
                Vec grad_temp = new Vec(gradients.len);
		updateGradient(x, gradients);             
                double p = (learning_rate * lambda-1);
              //  int q=  weights.size() - mLayers.get(mLayers.size() - 1).getWeightSize();
                int q=202;
              //System.out.println("weights before "+weights.toString());       
                for(int i = q; i < weights.len; i++ ){
                        double xx = (weights.get(i)*(1-learning_rate * lambda));                        
                        weights.set(i, xx);                       
                }   
		weights.addScaled(learning_rate, gradients);
	} 
        
        void L1Regularization(Vec x, Vec y, double learning_rate, double lambda) {
        predict(x);
            backprop(weights, y);
            Vec gradients = new Vec(weights.len);
            updateGradient(x, gradients);  
            
            double p = (learning_rate * lambda-1); 
            double xx;
            int q=202;
            for(int i = q; i < weights.len; i++ ){
                if(weights.get(i)>0){
                        xx = (weights.get(i)-(learning_rate * lambda));                        
                        weights.set(i, xx);                       
                } 
                else {
                    xx = (weights.get(i)+(learning_rate * lambda));                        
                        weights.set(i, xx);                       
                }                 
        }
        weights.addScaled(learning_rate, gradients);
    }
}
