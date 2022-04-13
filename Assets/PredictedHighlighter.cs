using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PredictedHighlighter : MonoBehaviour
{
    
    public Material highlightMaterial;
    private Material previousMaterial;
    private IntentionPredictor predictor;
    
    void Start()
    {
        predictor = GetComponent<IntentionPredictor>();
    }

    //Changes the material of the object to the highlight material
    public void Highlight(GameObject objectToHighlight)
    {
        if(previousMaterial != null)
            objectToHighlight.GetComponent<Renderer>().material = previousMaterial;

        if (objectToHighlight != null)
        {
            previousMaterial = objectToHighlight.GetComponent<Renderer>().material;
            objectToHighlight.GetComponent<Renderer>().material = highlightMaterial;
        }
        
    }
    
}
