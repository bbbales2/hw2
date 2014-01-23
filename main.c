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

double ddot(int n, double *v, double *vv)
{
  double total = 0.0;

  for(int i = 0; i < n; i++)
    total += v[i] * vv[i];

  return total;
}

double daxpy(int n, double a, double *x, double *y)
{
  for(int i = 0; i < n; i++)
    y[i] += a * x[i];
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

    b = (double *)malloc(sizeof(double) * n);
	
    for(int i = 0; i < n; i++)
      b[i] = (double)i;//cs240_getB(i, n);
    
    // each processor calls cs240_getB to build its own part of the b vector!
  } else if  ( !strcmp( argv[1], "-i" ) && argc == 4 ) {
    exit(-1);
    b = load_vec( argv[2], &k );
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

  printf("Hi from rank %d of %d, z = %d\n", rank, size, z);
	
  double *x = (double *)malloc(sizeof(double) * n),
    *r = (double *)malloc(sizeof(double) * n),
    *d = (double *)malloc(sizeof(double) * n),
    *dz = (double *)malloc(sizeof(double) * z),
    *xz = (double *)malloc(sizeof(double) * z),
    *rz = (double *)malloc(sizeof(double) * z),
    *ad = (double *)malloc(sizeof(double) * n),
    *adz = (double *)malloc(sizeof(double) * z),
    *below = (double *)malloc(sizeof(double) * k),
    *bz = (double *)malloc(sizeof(double) * z),    
    *total = (double *)malloc(sizeof(double) * size),   //Place to add up all individual dot products
    *above = (double *)malloc(sizeof(double) * k);

    double totalz = 0.0;
    MPI_Status error;

  memset(x, 0, sizeof(double) * n);
  memcpy(r, b, sizeof(double) * n);
  memcpy(d, r, sizeof(double) * n);

//////Send x,r,b and d to processors in xz, rz, bz and dz to initialize loop
      if(rank == 0)
        {
          for(int i = 1; i < size; i++)
            {
              //Send a block of x to each processor
              MPI_Send(&x[i * z], z, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
          //Copy components from x to local vector xz
          memcpy(xz, x, sizeof(double) * z);
        }
      else
        {
          //printf("%d is receiving block of size %d\n", rank, z * k);
          MPI_Recv(xz, z, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &error);
        }

      if(rank == 0)
          {
            for(int i = 1; i < size; i++)
              {
                //Send a block of r to each processor
                MPI_Send(&r[i * z], z, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
              }
            //Copy components from r to local vector rz
            memcpy(rz, r, sizeof(double) * z);
          }
      else
          {
            //printf("%d is receiving block of size %d\n", rank, z * k);
            MPI_Recv(rz, z, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &error);
          }

      if(rank == 0)
        {
          for(int i = 1; i < size; i++)
            {
              //Send a block of b to each processor
              MPI_Send(&b[i * z], z, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
          //Copy components from b to local vector bz
          memcpy(bz, b, sizeof(double) * z);
        }
      else
        {
          //printf("%d is receiving block of size %d\n", rank, z * k);
          MPI_Recv(bz, z, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &error);
        }

      if(rank == 0)
        {
          for(int i = 1; i < size; i++)
            {
              //printf("Sending block %d of size %d %lx\n", i, z * k, (unsigned long int)d);
              MPI_Send(&d[i * z], z, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }

          memcpy(dz, d, sizeof(double) * z);
        }
      else
        {
          //printf("%d is receiving block of size %d\n", rank, z * k);
          MPI_Recv(dz, z, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &error);
        }

//////////CALCULATE RTR VIA DDOT (parallel)
    double rtr = 0.0;//= ddot(n, r, r);

    for(int i = 0; i < z; i++)
      {
        totalz += rz[i] * rz[i];
      }
    
    //Collect sums back up
    if(rank == 0)
      {
        for(int i = 1; i < size; i++)
          {
            MPI_Recv(&total[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &error); 
          }
  
        total[0]=totalz;
      }
    else
      {
        MPI_Send(&totalz, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }
      
    for(int i = 0; i < size; i++)
      {
        rtr += total[i];
      }

  double norm = 1;
//////////CALCULATE NORMB VIA DDOT (parallel)
  double normb =0.0;//= sqrt(ddot(n, b, b));
    
    totalz=0;

    for(int i = 0; i < z; i++)
      {
        totalz += bz[i] * bz[i];
      }

    //Collect sums back up
    if(rank == 0)
      {
        for(int i = 1; i < size; i++)
          {
            MPI_Recv(&total[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &error); 
          }
  
        total[0]=totalz;
      }
    else
      {
        MPI_Send(&totalz, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }
      
    for(int i = 0; i < size; i++)
      {
        normb += total[i];
      }

  normb = sqrt(normb);

  int niters = 0;

  while(norm > 1e-6 && niters < iterations || rank > 0)
    {
      niters = niters + 1;

      memset(ad, 0, sizeof(double) * n);
      memset(adz, 0, sizeof(double) * z);

      int stat = 1;

      MPI_Bcast(&stat, 1, MPI_INT, 0, MPI_COMM_WORLD);

      if(stat != 1)
        break;

      //printf("Distributing d\n");

      // Each processor does their subblock
      //   Needs last row from neighbor below and first row from neighbor above

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
 
      //////////////COMPUTE ALPHA VIA DDOT (parallel)
      double alpha = 0.0;//rtr / ddot(n, d, ad);

      totalz=0;

      for(int i = 0; i < z; i++)
        {
          totalz += dz[i] * adz[i];
        }

      //Collect sums back up
      if(rank == 0)
        {
          for(int i = 1; i < size; i++)
            {
              MPI_Recv(&total[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &error); 
            }
  
          total[0]=totalz;
        }
      else
        {
          MPI_Send(&totalz, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        } 
      
      for(int i = 0; i < size; i++)
        {
          alpha += total[i];
        }

      alpha = rtr/alpha;

      MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


      //daxpy(n, alpha, d, x);
      //daxpy(n, -alpha, ad, r);

///////////////Parallel daxpy
      for(int i = 0; i < z; i++)
        {
          xz[i] += alpha * dz[i];
          rz[i] -= alpha * adz[i];
        }

      double rtrold = rtr;
      //////////CALCULATE RTR VIA DDOT (parallel)
      rtr = 0.0;//ddot(n, r, r);
     
      totalz=0;

      for(int i = 0; i < z; i++)
        {
          totalz += rz[i] * rz[i];
        }
    
      //Collect sums back up
      if(rank == 0)
        {
          for(int i = 1; i < size; i++)
            {
              MPI_Recv(&total[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &error); 
            }
  
          total[0]=totalz;
        }
      else
        {
          MPI_Send(&totalz, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
      
      for(int i = 0; i < size; i++)
        {
          rtr += total[i];
        }

      double beta = rtr / rtrold;

      MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

////////////Compute d=r+beta*d in parallel
      for(int i = 0; i < z; i++)
        {
          dz[i] = rz[i] + beta * dz[i];
        }

      if(rank != 0)
        {
          continue;
        }

      norm = sqrt(rtr) / normb;

      if(rank == 0)
        {
          printf("%f\n", norm);
        }
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
      if ( writeOutX ) {
        save_vec( k, x );
      }
      
      // Output
      printf( "Problem size (k): %d\n",k);
      printf( "Norm of the residual after %d iterations: %lf\n",iterations,norm);
      printf( "Elapsed time during CGSOLVE: %lf\n", t2-t1);
    }
  // Deallocate 
  free(above);
  free(below);
  free(b);
  free(x);
  free(r);
  free(d);
  free(dz);
  free(ad);
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
