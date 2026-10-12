import math

import pygame


class ParticleDisplay:
    def __init__(self, scanner, unit=28):
        self.scanner = scanner
        self.unit = unit
        self.closed = False
        self.map_width = scanner.map_cols * unit
        self.map_height = scanner.map_rows * unit
        self.hud_height = 54

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.map_width, self.map_height + self.hud_height)
        )
        pygame.display.set_caption("VP718 Lab 4 - Particle Filter")
        self.font = pygame.font.Font(None, 19)
        self.small_font = pygame.font.Font(None, 17)
        self.background = self._make_background()

    def _make_background(self):
        background = pygame.Surface(self.screen.get_size())
        background.fill((18, 23, 29))
        pygame.draw.rect(
            background,
            (25, 31, 38),
            (0, 0, self.map_width, self.map_height),
        )

        for row in range(self.scanner.map_rows):
            for column in range(self.scanner.map_cols):
                grid_x = column * self.scanner.scale
                grid_y = row * self.scanner.scale
                if self.scanner.grid[grid_y, grid_x] == 1:
                    rect = (
                        column * self.unit,
                        (self.scanner.map_rows - row - 1) * self.unit,
                        self.unit,
                        self.unit,
                    )
                    pygame.draw.rect(background, (111, 125, 137), rect)

        for column in range(self.scanner.map_cols + 1):
            x = column * self.unit
            pygame.draw.line(
                background,
                (36, 43, 50),
                (x, 0),
                (x, self.map_height),
                1,
            )
        for row in range(self.scanner.map_rows + 1):
            y = row * self.unit
            pygame.draw.line(
                background,
                (36, 43, 50),
                (0, y),
                (self.map_width, y),
                1,
            )
        return background

    def to_pixels(self, x, y):
        return (
            round(x * self.unit),
            round((self.scanner.map_rows - y) * self.unit),
        )

    def _draw_pose(self, pose, colour):
        if pose is None:
            return

        x, y, heading = pose
        forward_x = math.cos(heading)
        forward_y = math.sin(heading)
        left_x = -forward_y
        left_y = forward_x
        points = (
            (x + forward_x * 0.48, y + forward_y * 0.48),
            (x - forward_x * 0.30 + left_x * 0.20,
             y - forward_y * 0.30 + left_y * 0.20),
            (x - forward_x * 0.30 - left_x * 0.20,
             y - forward_y * 0.30 - left_y * 0.20),
        )
        pixels = [self.to_pixels(point_x, point_y) for point_x, point_y in points]
        pygame.draw.polygon(self.screen, colour, pixels)
        pygame.draw.polygon(self.screen, (12, 16, 20), pixels, width=1)

    def _draw_legend(self, particle_filter, ess):
        hud_y = self.map_height
        pygame.draw.rect(
            self.screen,
            (18, 23, 29),
            (0, hud_y, self.map_width, self.hud_height),
        )
        entries = (
            ((44, 194, 211), "Particles"),
            ((240, 177, 72), "Odometry"),
            # ((229, 91, 164), "PF estimate"),
            ((105, 211, 128), "Ground truth"),
        )
        x = 12
        for colour, label in entries:
            pygame.draw.circle(self.screen, colour, (x + 5, hud_y + 15), 4)
            text = self.small_font.render(label, True, (218, 225, 231))
            self.screen.blit(text, (x + 14, hud_y + 8))
            x += text.get_width() + 33

        status = self.font.render(
            f"ESS {ess:.1f} / {particle_filter.num_particles}",
            True,
            (222, 231, 237),
        )
        self.screen.blit(status, (12, hud_y + 30))

    def draw(self, particle_filter, odometry_pose, estimate_pose, ground_truth_pose, ess):
        if self.closed:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        self.screen.blit(self.background, (0, 0))
        for particle in particle_filter.particles:
            point = self.to_pixels(particle[0], particle[1])
            pygame.draw.circle(self.screen, (44, 194, 211), point, 2)

        self._draw_pose(odometry_pose, (240, 177, 72))
        # self._draw_pose(estimate_pose, (229, 91, 164))
        self._draw_pose(ground_truth_pose, (105, 211, 128))
        self._draw_legend(particle_filter, ess)
        pygame.display.flip()
        return True

    def close(self):
        if not self.closed:
            pygame.display.quit()
            self.closed = True