#include <stdio.h>
#include <math.h>

#define PI 3.141592653589793238
#define FIELD_SIZE_X 1000
#define FIELD_SIZE_Y 1000

double Gaussian_func(double x, double x0, double y, double y0, double sx, double sy)
{
  double k;
  double z1;
  double z2;
  k = 1000*(1.0/(2*PI*sx*sy));
  z1 = -0.5*((x-x0)/sx)*((x-x0)/sx);
  z2 = -0.5*((y-y0)/sy)*((y-y0)/sy);
  return(k*exp(z1 + z2));
}

double dGaussian_func_dx(double x, double x0, double y, double y0, double sx, double sy)
{
  double k;
  double z1;
  double z2;
  k = 1000*(-1.0/(2*PI*(sx*sx*sx)*sy))*(x-x0);
  z1 = -0.5*((x-x0)/sx)*((x-x0)/sx);
  z2 = -0.5*((y-y0)/sy)*((y-y0)/sy);
  return(k*exp(z1 + z2));
}

double dGaussian_func_dy(double x, double x0, double y, double y0, double sx, double sy)
{
  double k;
  double z1;
  double z2;
  k = 1000*(-1.0/(2*PI*sx*(sy*sy*sy)))*(y-y0);
  z1 = -0.5*((x-x0)/sx)*((x-x0)/sx);
  z2 = -0.5*((y-y0)/sy)*((y-y0)/sy);
  return(k*exp(z1 + z2));
}

double func(double x, double y, double *x0, double *y0, double *sx, double *sy, int point_number)
{
  int i;
  double s = 0.0;
  for(i=0;i<point_number;i++)
  {
    s += -1.0*Gaussian_func(x, x0[i], y, y0[i], sx[i], sy[i]);
  }
  return(s);
}

double dfunc_dx(double x, double y, double *x0, double *y0, double *sx, double *sy, int point_number)
{
  int i;
  double s = 0.0;
  for(i=0;i<point_number;i++)
  {
    s += -1.0*dGaussian_func_dx(x, x0[i], y, y0[i], sx[i], sy[i]);
  }
  return(s);
}

double dfunc_dy(double x, double y, double *x0, double *y0, double *sx, double *sy, int point_number)
{
  int i;
  double s = 0.0;
  for(i=0;i<point_number;i++)
  {
    s += -1.0*dGaussian_func_dy(x, x0[i], y, y0[i], sx[i], sy[i]);
  }
  return(s);
}

void out_field_data(char *file_name, double *x, double *y, double *x0, double *y0, double *sx, double *sy, int point_number)
{
  FILE *fp;
  int i;
  int j;

  fp = fopen(file_name, "w");
  
  for(i=0;i<FIELD_SIZE_X;i++)
  {
    for(j=0;j<FIELD_SIZE_Y;j++)
    {
      fprintf(fp, "%f %f %f \n", x[i], y[j], func(x[i], y[j], x0, y0, sx, sy, point_number));
    }
    fprintf(fp, "\n");
  }
  
  fclose(fp);
}

int main(void)
{
  FILE *fp;
  
  int i;
  int j;
  int count;
  int point_number = 4;

  double x;
  double y;
  double z;

  double dx = 0.1;
  double dy = 0.1;
  
  double x0[4] = {100.0, 100.0, 130.0, 150.0};
  double y0[4] = {105.0, 150.0, 130.0, 105.0};
  double sx[4] = {13.0, 10.0, 10.0, 12.0};
  double sy[4] = {13.0, 10.0, 10.0, 12.0};

  //INITIAL PARAMETER
  //BEGIN
  double x_initial = 160.0;
  double y_initial = 160.0;
  //END
  
  double z_before;
  double df_dx;
  double df_dy;
  
  double step_size = 1.0;
  double cutoff = 1.0e-17;

  double x_plot[FIELD_SIZE_X];
  double y_plot[FIELD_SIZE_Y];

  x = 75.0;
  for(i=0;i<FIELD_SIZE_X;i++)
  {
    x_plot[i] = x;
    x += dx;
  }

  y = 80.0;
  for(j=0;j<FIELD_SIZE_Y;j++)
  {
    y_plot[j] = y;
    y += dy;
  }
  
  out_field_data("field.dat", x_plot, y_plot, x0, y0, sx, sy, point_number);

  count = 0;
  x = x_initial;
  y = y_initial;
  z = func(x_initial, y_initial, x0, y0, sx, sy, point_number);

  fp = fopen("data.dat", "w");
  fprintf(fp, "%d %.17f %.17f %.17f \n", count+1, x, y, z);
  fprintf(fp, "\n\n");
  while(1)
  {
    z_before = z;
    
    df_dx = dfunc_dx(x, y, x0, y0, sx, sy, point_number);
    df_dy = dfunc_dy(x, y, x0, y0, sx, sy, point_number);

    //printf("%f, %f\n", df_dx, df_dy);
    
    x -= step_size*df_dx;
    y -= step_size*df_dy;

    z = func(x, y, x0, y0, sx, sy, point_number);

    if(count%100==0)
    {
      printf("n = %d, x = %.17f, y = %.17f, z = %.17f \n", count+1, x, y, z);
      fprintf(fp, "%d %.17f %.17f %.17f \n", count+1, x, y, z);
    }
    if(fabs(z - z_before)<cutoff)
    {
      break;
    }
    count++;
  }
  //printf("\n\n");
  //printf("n = %d, x = %.17f, y = %.17f, z = %.17f \n", count+1, x, y, z);
  fprintf(fp, "\n\n");
  fprintf(fp, "%d %.17f %.17f %.17f \n", count+1, x, y, z);
  fclose(fp);
  
  return 0;
}
