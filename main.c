/* UCSB CS240A, Winter Quarter 2014
 * Main and supporting functions for the Conjugate Gradient Solver on a 5-point stencil
 *
 * NAMES:
 * PERMS:
 * DATE:
 */
#include "mpi.h"
#include "hw2harness.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double* load_vec( char* filename, int* k );
void save_vec( int k, double* x );

MPI_Status error;

double ddot(int z, double *v, double *vv)
{
  double total = 0.0;

  for(int i = 0; i < z; i++)
    total += v[i] * vv[i];

  double ttotal = 0.0;

  MPI_Reduce(&total, &ttotal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return ttotal;
}

double daxpy(int z, double alpha, double beta, double *x, double *y)
{
  for(int i = 0; i < z; i++)
    y[i] = beta * y[i] + alpha * x[i];
}

void Atimes(int rank, int size, int k, int z, double *dz, double *adz)
{
  double *above = (double *)malloc(sizeof(double) * k),
    *below = (double *)malloc(sizeof(double) * k);

  MPI_Request sendAbove, sendBelow, aboveRequest, belowRequest;

  memset(above, 0, sizeof(double) * k);
  memset(below, 0, sizeof(double) * k);

  if(rank != 0) {
    MPI_Irecv(above, k, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &aboveRequest);
    MPI_Isend(dz, k, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &sendAbove);
  }

  if(rank != size - 1)
    {
      MPI_Irecv(below, k, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &belowRequest);
      MPI_Isend(&dz[z - k], k, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &sendBelow);
    }

  if(rank != 0)
    {
      MPI_Wait(&aboveRequest, &error);
      MPI_Wait(&sendAbove, &error);
    }

  if(rank != size - 1)
    {
      MPI_Wait(&belowRequest, &error);
      MPI_Wait(&sendBelow, &error);
    }

  for(int i = 0; i < z; i++)
    {
      int j = rank * k + i / k, l = i % k;

      adz[i] = 4 * dz[i];

      if(j != 0)
        {
          if(i / k == 0)
            {
              adz[i] -= above[l];
            }
          else
            adz[i] -= dz[i - k];
        }

      if(j != k - 1)
        {
          if(i / k == (z / k - 1))
            {
              adz[i] -= below[l];
            }
          else
            adz[i] -= dz[i + k];
        }

      if(l != 0)
        adz[i] -= dz[i - 1];
          
      if(l != k - 1)
        adz[i] -= dz[i + 1];
    }

  free(above);
  free(below);
}

int main( int argc, char* argv[] ) {
  int writeOutX = 0;
  int n, k;
  int iterations = 1000;
  double *b, time;
  double t1, t2;
  int size, rank;
	
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // Read command line args.
  // 1st case runs model problem, 2nd Case allows you to specify your own b vector
  if ( argc == 3 ) {
    k = atoi( argv[1] );
    n = k*k;
  } else {
    printf( "\nCGSOLVE Usage: \n\t"
            "Model Problem:\tmpirun -np [number_procs] cgsolve [k] [output_1=y_0=n]\n\t"
            "Custom Input:\tmpirun -np [number_procs] cgsolve -i [input_filename] [output_1=y_0=n]\n\n");
    exit(0);
  }
  writeOutX = atoi( argv[argc-1] ); // Write X to file if true, do not write if unspecified.

  //z = the amount of data to be stored on each processor
  int z = n / size;
	
  // Start Timer
  t1 = MPI_Wtime();

  //printf("Hi from rank %d of %d, z = %d\n", rank, size, z);
	
  double *dz = (double *)malloc(sizeof(double) * z),
    *xz = (double *)malloc(sizeof(double) * z),
    *rz = (double *)malloc(sizeof(double) * z),
    *adz = (double *)malloc(sizeof(double) * z),
    *bz = (double *)malloc(sizeof(double) * z);

  double totalz = 0.0;
	
  // each processor calls cs240_getB to build its own part of the b vector!
  for(int i = 0; i < size; i++)
    for(int j = 0; j < z; j++)
      bz[j] = cs240_getB(i * z + j, n);

  memset(xz, 0, sizeof(double) * z);
  memcpy(rz, bz, sizeof(double) * z);
  memcpy(dz, rz, sizeof(double) * z);

  double rtr = ddot(z, rz, rz);

  double norm = 1;
//////////CALCULATE NORMB VIA DDOT (parallel)
  double normb = sqrt(ddot(z, bz, bz));
    
  int niters = 0;

  while(norm > 1e-6 && niters < iterations)
    {
      niters = niters + 1;

      memset(adz, 0, sizeof(double) * z);

      int stat = 1;

      MPI_Bcast(&stat, 1, MPI_INT, 0, MPI_COMM_WORLD);

      if(stat != 1)
        break;

      Atimes(rank, size, k, z, dz, adz);

      //printf("Distributing d\n");

      // Each processor does their subblock
      //   Needs last row from neighbor below and first row from neighbor above

 
      //////////////COMPUTE ALPHA VIA DDOT (parallel)
      double alpha = rtr / ddot(z, dz, adz);

      MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      ///////////////Parallel daxpy
      daxpy(z, alpha, 1.0, dz, xz);
      daxpy(z, -alpha, 1.0, adz, rz);

      double rtrold = rtr;
      //////////CALCULATE RTR VIA DDOT (parallel)
      rtr = ddot(z, rz, rz);
      double beta = rtr / rtrold;

      MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      ////////////Compute d=r+beta*d in parallel
      daxpy(z, 1.0, beta, rz, dz);

      if(rank != 0)
        {
          continue;
        }

      norm = sqrt(rtr) / normb;

      /*if(rank == 0)
        {
          printf("%f\n", norm);
          }*/
    }

  if(rank == 0)
    {
      int stat2 = 0;
      
      MPI_Bcast(&stat2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

  // End Timer
  t2 = MPI_Wtime();
	
  if(rank == 0)
    {
      //if ( writeOutX ) {
      //  save_vec( k, x );
      //}
      
      // Output
      //printf( "Problem size (k): %d\n",k);
      //printf( "Norm of the residual after %d iterations: %lf\n",iterations,norm);
      //printf( "Elapsed time during CGSOLVE: %lf\n", t2-t1);
      printf("%f %f\n", norm, t2 - t1);
    }
  // Deallocate 
  free(bz);
  free(xz);
  free(rz);
  free(dz);
  free(adz);
	
  MPI_Finalize();
	
  return 0;
}


/*
 * Supporting Functions
 *
 */

// Load Function
// NOTE: does not distribute data across processors
double* load_vec( char* filename, int* k ) {
  FILE* iFile = fopen(filename, "r");
  int nScan;
  int nTotal = 0;
  int n;
	
  if ( iFile == NULL ) {
    printf("Error reading file.\n");
    exit(0);
  }
	
  nScan = fscanf( iFile, "k=%d\n", k );
  if ( nScan != 1 ) {
    printf("Error reading dimensions.\n");
    exit(0);
  }
	
  n = (*k)*(*k);
  double* vec = (double *)malloc( n * sizeof(double) );
	
  do {
    nScan = fscanf( iFile, "%lf", &vec[nTotal++] );
  } while ( nScan >= 0 );
	
  if ( nTotal != n+1 ) {
    printf("Incorrect number of values scanned n=%d, nTotal=%d.\n",n,nTotal);
    exit(0);
  }
	
  return vec;
}

// Save a vector to a file.
void save_vec( int k, double* x ) { 
  FILE* oFile;
  int i;
  oFile = fopen("xApprox.txt","w");
	
  fprintf( oFile, "k=%d\n", k );
	
  for (i = 0; i < k*k; i++) { 
    fprintf( oFile, "%lf\n", x[i]);
  } 

  fclose( oFile );
}
