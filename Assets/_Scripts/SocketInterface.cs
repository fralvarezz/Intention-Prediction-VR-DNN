using System;
using System.Net.Sockets;
using System.Xml.Serialization;
using UnityEngine;

namespace _Scripts
{
    public class SocketInterface : MonoBehaviour
    {

        private TcpClient client;
        private NetworkStream stream;
        private bool isConnected;


        private void OnDestroy()
        {
            stream.Close();
            client.Close();
        }
        
        public void Connect(string server, String message)
        {
            try
            {
                Int32 port = 18500;
                client = new TcpClient(server, port);

                //Byte[] data = System.Text.Encoding.ASCII.GetBytes(message);

                stream = client.GetStream();
                isConnected = true;
            }
            catch (ArgumentNullException e)
            {
                Debug.Log(e);
            }
            catch (SocketException e)
            {
                Debug.Log(e);
            }
            
        }

        public void SendString(string message)
        {
            Byte[] data = System.Text.Encoding.ASCII.GetBytes(message);
            stream.Write(data, 0, data.Length);
        }

        public void SendFrame(float[] frame)
        {
            if (!isConnected)
                return;
            
            //XmlSerializer serializer = new XmlSerializer(typeof(float[]));

            var byteArray = new byte[frame.Length * 4];
            Buffer.BlockCopy(frame, 0, byteArray, 0, byteArray.Length);
            stream.Write(byteArray, 0, byteArray.Length);
        }
        
        
        public bool IsConnected()
        {
            return isConnected;
        }
    }
}