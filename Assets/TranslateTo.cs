using System.Collections;
using System.Collections.Generic;
using JetBrains.Annotations;
using UnityEngine;

public class TranslateTo : MonoBehaviour
{
    public Transform OffsetObjPos;

    public Transform OriginalObjPos;

    public Transform HaRT_coreObjPos;
    // Start is called before the first frame update
    void Start()
    {
        
        
    }

    void UpdatePos()
    {
        //Rotation
        //transform.rotation = OffsetObjPos.rotation;
        transform.rotation = Quaternion.Euler(0, OffsetObjPos.eulerAngles.y, 0);

        //Position

        var offset = OriginalObjPos.position - OffsetObjPos.position;
        
        transform.position -= offset;
        
        //Update HaRT_core position
        HaRT_coreObjPos.position -= offset;
        
        //Rotation HaRT_core rotation
        HaRT_coreObjPos.rotation = Quaternion.Euler(0, OffsetObjPos.eulerAngles.y, 0);

    }
    
    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown(KeyCode.R))
            UpdatePos();    
    }
}
