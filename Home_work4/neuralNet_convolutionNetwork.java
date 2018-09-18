
import java.util.ArrayList;
import java.util.Random;
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author pankaj
 */
public class neuralNet_convolutionNetwork extends SupervisedLearner{
    ArrayList<Layer> mLayers = new ArrayList<>();
    //ArrayList<Vec> mWeights = new ArrayList<>();
    Vec weights;
    int layerWeightSize;
    int nWeight_len;
    int filtersize1 = (5*5+1)*4;
    int filtersize2 = (3*3*4+1)*6;
    @Override
    String name() {
        return "neuralNet_convolutionNetwork";
    }
    
@Override
    void train(Matrix features, Matrix labels) {    
        mLayers.add(new LayerConv(new int[] {8, 8}, new int[]{5, 5}, new int[]{8, 8}, 4));
        mLayers.add(new LeakyRectifer(8 * 8 * 4));
        mLayers.add(new LayerMaxPooling2D(8, 8, 4));
        mLayers.add(new LayerConv(new int[]{4, 4, 4}, new int[]{3, 3, 4}, new int[]{4, 4, 1}, 6));
        mLayers.add(new LeakyRectifer(4 * 4 * 6));
        mLayers.add(new LayerMaxPooling2D(4, 4, 1 * 6));
        mLayers.add(new LayerLinear(2 * 2 * 6, 3));
        
        int layerCount = mLayers.size();
        int weight_linear = 0;
        int weight_conv = 0;
        int weight_ohter =0;
        layerWeightSize = 0;
        for(int i=0; i<layerCount; i++) {
            if(i==(layerCount-1)) {
                weight_linear =mLayers.get(i).mInput_size*mLayers.get(i).mOutput_size + mLayers.get(i).mOutput_size;
                layerWeightSize += weight_linear;
            }
            else if(i==0) {
                weight_conv =filtersize1;
                layerWeightSize += weight_conv;
            }
            else if(i==3) {
                weight_conv =filtersize2;
                layerWeightSize += weight_conv;
            }     
            else {
                weight_ohter = 0;
                layerWeightSize += weight_ohter;
            }            
        }
        weights = new Vec(layerWeightSize);
        initWeights();
    }
    
    void initWeights() {
        Random random = new Random();
        
        for (int i = 0; i < layerWeightSize; i = i + 1) {
           int mFilterElements = 5*5*4;
            double kk= random.nextGaussian()/mFilterElements;
                weights.set(i, kk);
              //  System.out.println("NeuralNet.initWeights()");
            }
        
    }

    @Override
    Vec predict(Vec in) {        
        Vec temp = new Vec(in, 0, in.len);
        int index = 0;
        for (int i = 0; i < mLayers.size(); i++) {
            Vec out_layer = new Vec(0);
            if ((i % 3) == 0) {
               if(i==0)
                    nWeight_len = filtersize1;
               if(i==3)
                    nWeight_len = filtersize2;
               if(i==6)
                    nWeight_len = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);
                    out_layer = new Vec(weights, index, nWeight_len);
                //System.out.println("out_layer size :  " +out_layer.len);
                index = index + nWeight_len;
            }
            mLayers.get(i).activate(out_layer, temp);
            temp = new Vec(mLayers.get(i).activation, 0, mLayers.get(i).activation.len);  
            //System.out.println("temp2 size :  " +temp.len);            
        }
        return temp;
    }
    
    
        
    
    void backprop(Vec weights, Vec target) {
        Vec layer_blame = new Vec(target, 0, target.len);
        Vec y_actual = mLayers.get(mLayers.size() - 1).activation; 
        y_actual.scale(-1.0);

        layer_blame.add(y_actual);
        int back_index = weights.len;
        //System.out.println("layer size :  " +mLayers.size());
        for (int i = mLayers.size() - 1; i >= 0; i--) {
            //System.out.println("loop count :  " +i); 
            if(i>0) {
                mLayers.get(i).blame.copy(layer_blame);
                Vec iweight = new Vec(0);
                int mod = i % 3;
                if (mod == 0) {
                    if(i==6)
                        nWeight_len = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);
                    else if(i==3)
                        nWeight_len = filtersize2;
                 //   else if(i==0)
                 //        nWeight_len = filtersize1;
                    else
                         nWeight_len = 0;

                    back_index = back_index - nWeight_len;
                    iweight = new Vec(weights, back_index, nWeight_len);
                }
                layer_blame = new Vec(mLayers.get(i - 1).mOutput_size);
                //System.out.println("previous Layer output size:  "+mLayers.get(i - 1).mOutput_size);
                //System.out.println("iweight:  "+iweight.toString());
                //System.out.println("iweight size:  "+iweight.size());
                //System.out.println("layer_blame size:  "+layer_blame.len);
                mLayers.get(i).backProp(iweight, layer_blame);
            }
            else
                mLayers.get(i).blame.copy(layer_blame);
        }
    }
    
        void updateGradient(Vec x, Vec gradient) {
        int grad_index = 0;

        for (int i = 0; i < mLayers.size(); i++) {
            int grad_size = 0;
            int mod = i % 3;
            //System.out.println("mod value :  " +mod);
            if (mod == 0) 
                if(i==0)
                    grad_size = filtersize1;
                if(i==3)
                    grad_size = filtersize2;
                if(i==6)
                    grad_size = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);            
                    
            Vec Grad_layer = new Vec(grad_size);

            if (i > 0) mLayers.get(i).updateGradient(mLayers.get(i - 1).activation, Grad_layer); 
            else  mLayers.get(i).updateGradient(x, Grad_layer);
                       
            for (int k = 0; k < grad_size; k++) {
                gradient.set(grad_index + k, Grad_layer.get(k));
            }
            if (mod == 0) {  
                grad_index += grad_size;
            }
            //System.out.println("updateGradient done\n");
        }
    }
    
    void refineWeights(Vec x, Vec y) {
        predict(x);
        backprop(weights, y);
        Vec gradient = new Vec(weights.len);
        updateGradient(x, gradient);
        //weights.addScaled(learningRate, gradient);
      //  System.out.println("gradients.len:  "+ gradient.len);
        
            //System.out.println("\n"+i+"   "+gradient.toString());
       //     System.out.println("gradient length    :" +gradient.len);
        
        Vec gradientVector = new Vec(weights.len);
        calculateFiniteDifferencing(x, y, weights, gradientVector);
        
        System.out.println("SL.   Gradients Vector from CNN--------------Gradients Vector from FiniteDifferencing");
        System.out.println("---   --------------------------             ----------------------------------------\n");
      //  System.out.println("\nGradients Vector from FiniteDifferencing\n");
        for(int i= 0;i<gradientVector.len;i++) {
            System.out.print((i+1)+".     "+gradient.get(i)+"                          "+gradientVector.get(i)+",  \n");
        }
        //System.out.println("gradient Vector length   "+gradientVector.len);
        double squred_error = gradientVector.squaredDistance(gradient);
        System.out.println("\n\nSquared Distance between CNN and FiniteDifferencing: " + squred_error);
    }
    
    void calculateFiniteDifferencing(Vec x, Vec y, Vec weights, Vec gradient) {
        double h = 0.0000001;        
        for (int r = 0; r<weights.len; r++) {            
            for(int j = 0; j < weights.len; j++){
                if(j!= r){
                    this.weights.set(j, weights.get(j));
                }
                else this.weights.set(j, weights.get(j)+h);
            }
            double sse1 = 0.0;
            Vec pred = predict(x);
            for(int j = 0; j < y.len; j++){
                sse1 += Math.pow(y.get(j) - (pred.get(j)), 2);
            }

            
            for(int j = 0; j < weights.len; j++){
                if(j!= r){
                    this.weights.set(j, weights.get(j));
                }
                else this.weights.set(j, weights.get(j)-h);
            }
            double sse2 = 0.0;
            Vec pred1 = predict(x);
            for(int j = 0; j < y.len; j++){
                sse2 += Math.pow(y.get(j) - (pred1.get(j)), 2);
            }
            
            double difference = (sse2-sse1)/(2.0*h);
            gradient.set(r, difference); 
            
           // System.out.println("Gradients Vector from FiniteDifferencing");
            //for(int i= 0;i<gradient.len;i++)
                //System.out.println("\n"+i+".  "+gradient.get(i));
        }
    }
}
