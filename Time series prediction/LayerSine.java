/**
 *
 * @author pankaj
 */
////////////////////////////////////////////////////////////////////////////////
public class LayerSine extends Layer {

   Vec temp;
    LayerSine(int inputs, int outputs){//, int outputs) {
        super(inputs, outputs);
        temp = new Vec(inputs);
    }
	
    @Override
    void activate(Vec weights, Vec x) {
        for(int i = 0; i < activation.len-1; i++){
            double act_val = (x.get(i));
            
            if(i<(activation.len-1)) {
                temp.set(i, x.get(i));
                activation.set(i, Math.sin(act_val));
            }
            else {
                temp.set(i, x.get(i));
                activation.set(i, x.get(i));
            }          
        } 
    }

    @Override
    void backprop(Vec weights, Vec prevBlame) {
        for(int i = 0; i < prevBlame.len; i++)	{
            double grad= activation.get(i);
            prevBlame.set(i, blame.get(i)* Math.cos(grad));	//d/dx(sinx)= cosx		
	}
    }
    @Override
    void updateGradient(Vec x, Vec gradient) {		
    }
}

