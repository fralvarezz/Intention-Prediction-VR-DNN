using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NatSuite.ML;
using NatSuite.ML.Features;
using NatSuite.ML.Hub;
using NatSuite.ML.Internal;


public class PredictExample : MonoBehaviour
{

    public MLModelData mlModelData;
    
    // Start is called before the first frame update
    void Start()
    {
        var model = mlModelData.Deserialize();
        Debug.Log(model);

        var predictor = new LSTMPredictor(model, mlModelData.labels);

        var inputs = new MLEdgeFeature();
        
        var (label, input) = predictor.Predict(inputs);
    }

    
    
    // Update is called once per frame
    void Update()
    {
        
    }
}
