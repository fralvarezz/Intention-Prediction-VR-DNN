using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawOnTexture : MonoBehaviour
{
    public Texture2D baseTexture;

    void Update()
    {
        DoMouseDrawing();
    }

    /// <summary>
    /// Allows drawing to the texture with a mouse
    /// </summary>
    /// <exception cref="Exception"></exception>
    private void DoMouseDrawing()
    {
        if (Camera.main == null)
        {
            throw new Exception("Can't find camera");
        }
        
        //is mouse pressed?
        if(!Input.GetMouseButton(0) && !Input.GetMouseButton(1)) return;

        Ray mouseRay = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;
        
        if(!Physics.Raycast(mouseRay, out hit)) return;

        if (hit.collider.transform != this.transform) return;

        Vector2 pixelUV = hit.textureCoord;
        pixelUV.x *= baseTexture.width;
        pixelUV.y *= baseTexture.height;

        Color colorToSet = Input.GetMouseButton(0) ? Color.white : Color.black;
        
        baseTexture.SetPixel((int)pixelUV.x, (int)pixelUV.y, colorToSet);
        baseTexture.Apply();
    }
}
