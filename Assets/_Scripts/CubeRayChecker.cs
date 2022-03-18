using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CubeRayChecker : MonoBehaviour
{
    private bool lookingAt;

    private MeshRenderer mr;
    private Material defaultMat;
    public Material lookAtMat;
    
    // Start is called before the first frame update
    void Start()
    {
        mr = GetComponent<MeshRenderer>();
        defaultMat = mr.material;
    }

    // Update is called once per frame
    void Update()
    {
        Recolor();
    }

    public void OnHit()
    {
        lookingAt = true;
    }

    public void OnHitLeft()
    {
        lookingAt = false;
    }
    
    private void Recolor()
    {
        if (lookingAt)
        {
            mr.material = lookAtMat;
        }
        else
        {
            mr.material = defaultMat;
        }
    }
    
}
