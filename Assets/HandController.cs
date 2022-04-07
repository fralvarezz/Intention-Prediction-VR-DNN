using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HandController : MonoBehaviour
{
    public GameObject collidingWith;
    

    private void OnTriggerEnter(Collider other)
    {
        if(!other.CompareTag("LookableObject"))
            return;
        
        collidingWith = other.gameObject;
    }

    private void OnTriggerExit(Collider other)
    {
        if(!other.CompareTag("LookableObject"))
            return;
        
        collidingWith = null;
    }
}
