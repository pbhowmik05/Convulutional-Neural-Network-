import java.util.ArrayList;
import java.util.List;


public class LayerMaxPooling2D extends Layer {
    List<Integer> indextobackPropagation = new ArrayList<>();
    int RowCount;
    int ColCount;
    int DepthCount;  

    LayerMaxPooling2D(int Col, int Row, int Dep) {
        super(Col*Row*Dep, Col*Row*Dep/4);
        
        ColCount = Col;
        RowCount = Row;
        DepthCount = Dep;
    }


    @Override
    void activate(Vec weights, Vec x) {
        Vec poolingSize = new Vec(RowCount*ColCount*DepthCount/4);
        int indexforRow = 0;
        int indexadded = 0;
        
        for (int i= 0; i< DepthCount*(RowCount/2); i++) {
            
            ArrayList<Vec> poolVecValue = new ArrayList<>();
            ArrayList<Integer> IndexToMaxVal = new ArrayList<>();
            
            for (int k= 0; k<2; k++) {	
            	poolVecValue.add(new Vec(x, indexforRow, this.ColCount));
            	IndexToMaxVal.add(indexforRow);
                indexforRow += this.ColCount;
            }

            int indexforColumn = 0;
            
            for (int j=0; j < ColCount; j += 2) {
                Vec temp_value = new Vec(poolVecValue.size()*2);  
                Vec temp_index = new Vec(poolVecValue.size()*2);
            	for (int k = 0; k < poolVecValue.size(); k++) {                   
                    Vec vec = poolVecValue.get(k);                   
                    for (int l = 0; l < 2; l++) {                 
                         temp_value.set((k*poolVecValue.size()+l), vec.get(indexforColumn+l));
                         temp_index.set((k*poolVecValue.size()+l), IndexToMaxVal.get(k) + indexforColumn + l);
                    }
                }
                double valMax = temp_value.get(0);
                int pointer = 0;
                for (int m = 1; m < poolVecValue.size()*2; m++){
                    if(valMax < temp_value.get(m)){
                        valMax = temp_value.get(m);
                        pointer = m;
                    }
                }
                int indexMax =  (int) temp_index.get(pointer);   
  
                poolingSize.set(indexadded, valMax);
                indextobackPropagation.add(indexMax);
                
                indexadded++;
                indexforColumn += 2;
            }
        }
        this.activation.copy(poolingSize);
    }

    @Override
    void backProp(Vec weights, Vec prevBlame) {
        for (int i=0; i<this.blame.size(); ++i) {
            prevBlame.set(indextobackPropagation.get(i), this.blame.get(i));
        }
    }

    @Override
    void updateGradient(Vec x, Vec gradient) {

    }
}