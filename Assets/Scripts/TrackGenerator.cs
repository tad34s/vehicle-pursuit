using EasyRoads3Dv3;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TrackGenerator : MonoBehaviour
{
    public GameObject marker;

    private SplineCreator sc;

    void Start()
    {
        sc = GetComponent<SplineCreator>();

        GenerateTrack();
        GameObject road = GameObject.FindGameObjectWithTag("Road");
        Mesh roadMesh = road.GetComponent<MeshFilter>().mesh;
        CreateMarkers(roadMesh);
    }

    private void GenerateTrack()
    {
        sc.GenerateVoronoi();
        sc.GenerateSpline();
        sc.CreateTrack();
    }

    private void CreateMarkers(Mesh roadMesh)
    {
        Vector3[] vertices = roadMesh.vertices;
        for (var i = 0; i < vertices.Length - 3; i += 2)
        {
            Vector3 markerPosition = new Vector3(
                (vertices[i].x + vertices[i + 3].x) / 2,
                0,
                (vertices[i].z + vertices[i + 3].z) / 2
            );
            Instantiate(marker, markerPosition, Quaternion.identity);
        }
    }

    // Update is called once per frame
    void Update()
    {
    }
}
