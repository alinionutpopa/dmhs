function [] = plotSkel3D( pose3D, colorOption )


ind = [1     2     3     4     7     8     9    13    14    15    16    18    19    20    26   27    28];
buff_large = zeros(32, 3);
buff_large(ind, :) = pose3D;

pose3D = buff_large';

hold on;
order = [1 3 2];
plot3(pose3D(order(1), [1 13]), pose3D(order(2), [1 13]), -pose3D(order(3), [1 13]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [13 14]), pose3D(order(2), [13 14]), -pose3D(order(3), [13 14]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [14 15]), pose3D(order(2), [14 15]), -pose3D(order(3), [14 15]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [16 15]), pose3D(order(2), [16 15]), -pose3D(order(3), [16 15]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [14 18]), pose3D(order(2), [14 18]), -pose3D(order(3), [14 18]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [18 19]), pose3D(order(2), [18 19]), -pose3D(order(3), [18 19]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [19 20]), pose3D(order(2), [19 20]), -pose3D(order(3), [19 20]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [14 26]), pose3D(order(2), [14 26]), -pose3D(order(3), [14 26]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [26 27]), pose3D(order(2), [26 27]), -pose3D(order(3), [26 27]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [27 28]), pose3D(order(2), [27 28]), -pose3D(order(3), [27 28]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [1 2]), pose3D(order(2), [1 2]), -pose3D(order(3), [1 2]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [2 3]), pose3D(order(2), [2 3]), -pose3D(order(3), [2 3]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [3 4]), pose3D(order(2), [3 4]), -pose3D(order(3), [3 4]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [1 7]), pose3D(order(2), [1 7]), -pose3D(order(3), [1 7]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [7 8]), pose3D(order(2), [7 8]), -pose3D(order(3), [7 8]), [colorOption '-'], 'LineWidth',5);
plot3(pose3D(order(1), [8 9]), pose3D(order(2), [8 9]), -pose3D(order(3), [8 9]), [colorOption '-'], 'LineWidth',5);

xlabel('X');
ylabel('Y')
zlabel('Z')
grid on

view(0,0);
axis equal;
