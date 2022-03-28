using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;

public class Predictor : MonoBehaviour
{

    public Texture2D texture;
    
    public NNModel modelAsset;

    private Model _runtimeModel;

    private IWorker _engine;

    /// <summary>
    /// A struct used for holding the results of our prediction in a way that's easy for us to view from the inspector.
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
        // Set up the runtime model and worker.
        _runtimeModel = ModelLoader.Load(modelAsset);
        _engine = WorkerFactory.CreateWorker(_runtimeModel, WorkerFactory.Device.GPU);
        // Instantiate our prediction struct.
        prediction = new Prediction();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            // making a tensor out of a grayscale texture
            var channelCount = 1; //grayscale, 3 = color, 4 = color+alpha
            // Create a tensor for input from the texture.
            print(_runtimeModel.layout);
            var inputX = new Tensor(texture, channelCount);
            inputX = inputX.Reshape(new TensorShape(1,1,28,28));
            
            var dummyAcceptedInput = new Tensor(1, 1, 28, 28); //random data that doesn't throw errors
            print(inputX.flatHeight);
            print(inputX.flatWidth);
            print(inputX.shape);
            
            print(dummyAcceptedInput.flatHeight);
            print(dummyAcceptedInput.flatWidth);
            print(dummyAcceptedInput.shape);

            // Peek at the output tensor without copying it.
            Tensor outputY = _engine.Execute(inputX).PeekOutput();
            // Set the values of our prediction struct using our output tensor.
            prediction.SetPrediction(outputY);
            
            // Dispose of the input tensor manually (not garbage-collected).
            inputX.Dispose();
        }
    }

    private void OnDestroy()
    {
        // Dispose of the engine manually (not garbage-collected).
        _engine?.Dispose();
    }
}