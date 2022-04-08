using System;
using NatSuite.ML;
using NatSuite.ML.Features;
using NatSuite.ML.Internal;

public class LSTMPredictor : IMLPredictor<(string label, float confidence)>
{

    public readonly string[] labels;
    private readonly MLEdgeModel model;
    
    public LSTMPredictor(MLModel model, string[] labels)
    {
        this.model = (MLEdgeModel) model;
        this.labels = labels;
    }
    
    public (string label, float confidence) Predict(params MLFeature[] inputs)
    {
        var modelInputType = this.model.inputs[0];
        using var inputFeature = (inputs[0] as IMLEdgeFeature).Create(modelInputType);
        using var outputFeatures = model.Predict(inputFeature);

        var arrayFeature = new MLArrayFeature<float>(outputFeatures[0]);
        return (arrayFeature[0].ToString(), 0.1f);
    }

    void IDisposable.Dispose() { }
}
