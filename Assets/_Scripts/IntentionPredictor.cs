using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using Unity.Barracuda;
using UnityEngine;

public class IntentionPredictor : MonoBehaviour
{
    public NNModel modelAsset;

    private Model runtimeModel;

    private IWorker worker;
    
    public List<GameObject> objects;
    private PredictedHighlighter highlighter;

    public TextMeshProUGUI textMeshProUGUI;
    public bool isTesting;

    [Button("PredictFromCSV", "Predict from CSV", false)]
    public string name;

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
            //Debug.Log($"Predicted {predictedValue}");
        }
    }

    public Prediction prediction;
    
    void Start()
    {
        highlighter = GetComponent<PredictedHighlighter>();
        
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

        if (isTesting)
            return;

        Debug.Log("Data was available!");
        float[,] input = EyeLogger.Instance.GetData();
        //print(input.GetLength(0));
        //print(input.GetLength(1));
        
        // TODO: I find unlikely that this worked first try
        Tensor inputTensor = new Tensor(new TensorShape(1,1,19,45), input);

        
        
        Tensor outputTensor = worker.Execute(inputTensor).PeekOutput();
        
        prediction.SetPrediction(outputTensor);
        HighlightPredictedObject();
        Debug.Log($"Predicted {GetPredictedObject()}");
        
        inputTensor.Dispose();
    }

    private void OnDestroy()
    {
        // Dispose of the engine manually (not garbage-collected).
        worker?.Dispose();
    }
    
    //Returns the predicted GameObject
    public GameObject GetPredictedObject()
    {
        //Debug.Log($"Predicted item with id {prediction.predictedValue}");
        return objects[prediction.predictedValue];
    }
    
    //Highlights the predicted GameObject
    public void HighlightPredictedObject()
    {
        highlighter.Highlight(GetPredictedObject());
        textMeshProUGUI.text = "Predicted:" + prediction.predictedValue.ToString();
    }

    public void SendFrameNetwork()
    {
        var data = CSVParser.ParseCSV();

        foreach (var output in data)
        {
            if (output.Count == 0)
                continue;
            foreach (var segment in output)
            {
                

                for (int i = 0; i < segment.GetLength(0); i++)
                {
                    var frame = new float[segment.GetLength(1)];
                    for (int j = 0; j < segment.GetLength(1); j++)
                    {
                        frame[j] = segment[i, j];
                        Debug.Log(frame);
                    }
                }
            }
        }
    }
    
    public void PredictFromCSV()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(runtimeModel, WorkerFactory.Device.GPU);
        
        // Instantiate the prediction struct.
        prediction = new Prediction();
        
        var data = CSVParser.ParseCSV(3, false);

        int numCorrect = 0;
        int numTotal = 0;
        foreach (var output in data)
        {
            if(output.Count == 0)
                continue;
            for (int i = 0; i < 2; i++)
            {
                var segment = output[i];
                int startingFrame = 0;
                int endingFrame = startingFrame + 45;
                float[,] input = new float[45, segment.GetLength(1) - 1];
                while (endingFrame < segment.GetLength(0))
                {
                    for (int j = startingFrame; j < endingFrame; j++)
                    {
                        for (int k = 0; k < segment.GetLength(1) - 1; k++)
                        {
                            input[j - startingFrame, k] = segment[j, k];
                        }
                    }
                    Tensor inputTensor = new Tensor(new TensorShape(1,1,19,45), input);

                    Tensor outputTensor = worker.Execute(inputTensor).PeekOutput();
                    prediction.SetPrediction(outputTensor);
                    //Debug.Log("Predicted " + prediction.predictedValue + ". Actual: " + segment[endingFrame, segment.GetLength(1) - 1]);
                    if(prediction.predictedValue == (int)segment[endingFrame, segment.GetLength(1) - 1])
                    {
                        numCorrect++;
                    }
                    numTotal++;

                    startingFrame += 1;
                    endingFrame = startingFrame + 45;
                    inputTensor.Dispose();
                    outputTensor.Dispose();
                }
            }
        }
        Debug.Log("Num correct: " + numCorrect + " out of " + numTotal);
        Debug.Log("Accuracy: " + ((float)numCorrect / (float)numTotal) * 100 + "%");
        
        /*float[,] input = new float[45, data[3][0].GetLength(1) - 1];
        
        for (int i = 0; i < 45; i++)
        {
            for (int j = 0; j < data[3][0].GetLength(1) - 1; j++)
            {
                input[i, j] = data[3][0][i, j];
            }
        }
        
        Tensor inputTensor = new Tensor(new TensorShape(1,1,19,45), input);

        Tensor outputTensor = worker.Execute(inputTensor).PeekOutput();
        
        prediction.SetPrediction(outputTensor);
        Debug.Log($"Predicted {GetPredictedObject()}");
        
        inputTensor.Dispose();
        outputTensor.Dispose();
        */     
    }
    
}
