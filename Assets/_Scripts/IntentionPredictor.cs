using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;

public class IntentionPredictor : MonoBehaviour
{
    public NNModel modelAsset;

    private Model runtimeModel;

    private IWorker worker;
    
    /// <summary>
    /// A struct used for holding the results of our prediction in a way that's easy to view from the inspector.
    /// </summary>
    [Serializable]
    public struct Prediction
    {
        // The most likely value for this prediction
        public int predictedValue;
        // The list of likelihoods for all the possible classes
        public float[] predicted;

        public void SetPrediction(Tensor t)
        {
            // Extract the float value outputs into the predicted array.
            predicted = t.AsFloats();
            // The most likely one is the predicted value.
            predictedValue = Array.IndexOf(predicted, predicted.Max());
            Debug.Log($"Predicted {predictedValue}");
        }
    }

    public Prediction prediction;
    
    void Start()
    {
        // Load the model and make a worker
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(runtimeModel, WorkerFactory.Device.GPU);
        
        // Instantiate the prediction struct.
        prediction = new Prediction();
    }

    void Update()
    {
        if(!EyeLogger.Instance.dataIsReady)
            return;
        
        Debug.Log("Data was available!");
        float[,] input = EyeLogger.Instance.GetData();
        //print(input.GetLength(0));
        //print(input.GetLength(1));
        
        // TODO: I find unlikely that this worked first try
        Tensor inputTensor = new Tensor(new TensorShape(1,1,19,1), input);

        Tensor outputTensor = worker.Execute(inputTensor).PeekOutput();
        
        prediction.SetPrediction(outputTensor);
        
        inputTensor.Dispose();
    }

    private void OnDestroy()
    {
        // Dispose of the engine manually (not garbage-collected).
        worker?.Dispose();
    }
}
