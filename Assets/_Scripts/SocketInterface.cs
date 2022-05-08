using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Xml.Serialization;
using UnityEngine;

namespace _Scripts
{
    public class SocketInterface : MonoBehaviour
    {
        private TcpClient client;
        private NetworkStream stream;
        private bool isConnected;
        private Thread recvThread;
        private bool attemptingConnection;
        private int latestPrediction = -9999;
        private float latestTimestamp = 0f;
        private DateTime latestTime;
        private void OnDestroy()
        {
            recvThread?.Abort();
            stream.Close();
            client.Close();
        }

        public void Connect(string server, Int32 port)
        {
            try
            {
                client = new TcpClient(server, port);
                stream = client.GetStream();
                isConnected = true;
                Debug.Log("Connected to server on " + server + ":" + port);

                StartListener();
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

        private void Update()
        {
            if (!isConnected && !attemptingConnection)
            {
                StartCoroutine(Reconnect("localhost", 18500));
                attemptingConnection = true;
            }
        }

        private IEnumerator Reconnect(string server, Int32 port)
        {
            while (!isConnected)
            {
                try
                {
                    client = new TcpClient(server, port);
                    stream = client.GetStream();
                    isConnected = true;
                    StartListener();
                }
                catch (SocketException e)
                {
                    isConnected = false;
                }

                yield return new WaitForSeconds(2);
            }
        }

        private void StartListener()
        {
            try
            {
                //if (recvThread.IsAlive)
                // {
                //     recvThread.Abort();
                //   }
                recvThread = new Thread(ListenForData);
                recvThread.IsBackground = true;
                recvThread.Start();
            }
            catch (Exception e)
            {
                Debug.LogError(e);
            }
        }

        public void SendString(string message)
        {
            Byte[] data = System.Text.Encoding.ASCII.GetBytes(message);
            stream.Write(data, 0, data.Length);
        }


        private void ListenForData()
        {
            Debug.Log("ListenForData() started on background thread");
            while (isConnected)
            {
                int length;
                byte[] buffer = new byte[4];
                try
                {
                    while ((length = stream.Read(buffer, 0, buffer.Length)) != 0)
                    {
                        var recvData = new byte[length];
                        Array.Copy(buffer, 0, recvData, 0, length);

                        int resp = BitConverter.ToInt32(recvData.Reverse().ToArray(), 0);
                        latestPrediction = resp;
                        latestTime = DateTime.Now;
                        Debug.Log("Message received: " + resp);
                    }
                }
                catch (Exception e)
                {
                    Debug.Log("Caught");
                    Debug.Log(e);
                    if (e.GetType() == typeof(IOException))
                    {
                        isConnected = false;
                        attemptingConnection = false;
                    }
                }
            }

            Debug.Log("ListenForData() terminating");
        }
        

        public void SendFrame(float[] frame)
        {
            if (!isConnected)
                return;
            var byteArray = new byte[frame.Length * 4];
            Buffer.BlockCopy(frame, 0, byteArray, 0, byteArray.Length);
            stream.Write(byteArray, 0, byteArray.Length);
        }


        public bool IsConnected()
        {
            return isConnected;
        }

        // Get latest prediction made by the network
        public int getPredition()
        {
            return latestPrediction;
        }
        // Get timestamp of when latest prediction was received by the network
        public DateTime getTimestamp()
        {
            return latestTime;
        }
    }
}