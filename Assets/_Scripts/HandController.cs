using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HandController : MonoBehaviour
{
    public GameObject collidingWith;
    

    private void OnTriggerEnter(Collider other)
    {
        collidingWith = other.gameObject;
    }

    private void OnTriggerExit(Collider other)
    {
        collidingWith = null;
    }
}
