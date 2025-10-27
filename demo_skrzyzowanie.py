import pygame
import random

SZEROKOSC, WYSOKOSC = 800, 600
KOLOR_TLA = (30, 30, 30)
KOLOR_DROGI = (100, 100, 100)
KOLOR_CZERWONY = (255, 0, 0)
KOLOR_ZIELONY = (0, 255, 0)
KOLOR_POJAZDU_EW = (200, 200, 0)
KOLOR_POJAZDU_NS = (0, 200, 200)

PAS_RUCHU_SZEROKOSC = 18
SRODEK_X = SZEROKOSC // 2
SRODEK_Y = WYSOKOSC // 2

STOP_LINIA_LEWA = SRODEK_X - PAS_RUCHU_SZEROKOSC * 2
STOP_LINIA_PRAWA = SRODEK_X + PAS_RUCHU_SZEROKOSC * 2
STOP_LINIA_GORNA = SRODEK_Y - PAS_RUCHU_SZEROKOSC * 2
STOP_LINIA_DOLNA = SRODEK_Y + PAS_RUCHU_SZEROKOSC * 2

SKRZYZOWANIE_BOX = pygame.Rect(
    STOP_LINIA_LEWA, 
    STOP_LINIA_GORNA, 
    STOP_LINIA_PRAWA - STOP_LINIA_LEWA, 
    STOP_LINIA_DOLNA - STOP_LINIA_GORNA
)

ODSTEP_POJAZDU = 15

class Pojazd:
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        
        if dx != 0:
            self.szerokosc = 30
            self.wysokosc = 15
            self.kolor = KOLOR_POJAZDU_EW
        else:
            self.szerokosc = 15
            self.wysokosc = 30
            self.kolor = KOLOR_POJAZDU_NS
            
        self.predkosc = 2
        self.stan = "JEDZIE"
        self.rect = pygame.Rect(self.x, self.y, self.szerokosc, self.wysokosc)

    def aktualizuj(self, skrzyzowanie, wszystkie_pojazdy):
        self.stan = "JEDZIE"

        przod_x = self.x + self.szerokosc + (self.dx * self.predkosc)
        przod_y = self.y + self.wysokosc + (self.dy * self.predkosc)

        if self.dx > 0:
            if przod_x >= STOP_LINIA_LEWA and self.x < STOP_LINIA_LEWA:
                if skrzyzowanie.stan_EW == "CZERWONE":
                    self.stan = "CZEKA_NA_SWIATLO"
                    
        elif self.dy > 0:
            if przod_y >= STOP_LINIA_GORNA and self.y < STOP_LINIA_GORNA:
                if skrzyzowanie.stan_NS == "CZERWONE":
                    self.stan = "CZEKA_NA_SWIATLO"

        if self.stan == "JEDZIE":
            for inny_pojazd in wszystkie_pojazdy:
                if inny_pojazd is self:
                    continue

                jest_na_tym_samym_pasie = False
                jest_bezposrednio_z_przodu = False

                if self.dx > 0:
                    jest_na_tym_samym_pasie = abs(self.y - inny_pojazd.y) < 5
                    odleglosc = inny_pojazd.x - (self.x + self.szerokosc)
                    jest_bezposrednio_z_przodu = (inny_pojazd.x > self.x) and (odleglosc < ODSTEP_POJAZDU)
                
                elif self.dy > 0:
                    jest_na_tym_samym_pasie = abs(self.x - inny_pojazd.x) < 5
                    odleglosc = inny_pojazd.y - (self.y + self.wysokosc)
                    jest_bezposrednio_z_przodu = (inny_pojazd.y > self.y) and (odleglosc < ODSTEP_POJAZDU)

                if jest_na_tym_samym_pasie and jest_bezposrednio_z_przodu:
                    self.stan = "CZEKA_NA_INNY_POJAZD"
                    break

        if self.stan == "JEDZIE":
            czy_wjezdzam = False
            if self.dx > 0 and przod_x >= STOP_LINIA_LEWA and self.x < STOP_LINIA_PRAWA:
                czy_wjezdzam = True
            elif self.dy > 0 and przod_y >= STOP_LINIA_GORNA and self.y < STOP_LINIA_DOLNA:
                czy_wjezdzam = True

            if czy_wjezdzam:
                for inny_pojazd in wszystkie_pojazdy:
                    if inny_pojazd is self:
                        continue
                    
                    if SKRZYZOWANIE_BOX.colliderect(inny_pojazd.rect):
                        self.stan = "CZEKA_NA_INNY_POJAZD"
                        break

        if self.stan == "JEDZIE":
            self.x += self.predkosc * self.dx
            self.y += self.predkosc * self.dy
            
        self.rect.topleft = (self.x, self.y)

    def rysuj(self, ekran):
        pygame.draw.rect(ekran, self.kolor, self.rect)


class KontrolerSkrzyzowania:
    def __init__(self):
        self.stan_EW = "ZIELONE"
        self.stan_NS = "CZERWONE"
        self.timer = 10 

    def aktualizuj(self, dt):
        self.timer -= dt
        if self.timer <= 0:
            if self.stan_EW == "ZIELONE":
                self.stan_EW = "CZERWONE"
                self.stan_NS = "ZIELONE"
                self.timer = 8 
            else:
                self.stan_EW = "ZIELONE"
                self.stan_NS = "CZERWONE"
                self.timer = 10 

    def rysuj(self, ekran):
        kolor_ew = KOLOR_ZIELONY if self.stan_EW == "ZIELONE" else KOLOR_CZERWONY
        pygame.draw.circle(ekran, kolor_ew, (STOP_LINIA_LEWA - 10, SRODEK_Y - 30), 10)

        kolor_ns = KOLOR_ZIELONY if self.stan_NS == "ZIELONE" else KOLOR_CZERWONY
        pygame.draw.circle(ekran, kolor_ns, (SRODEK_X - 30, STOP_LINIA_GORNA - 10), 10)
        
    def narysuj_drogi(self, ekran):
        pygame.draw.rect(ekran, KOLOR_DROGI, (0, SRODEK_Y - PAS_RUCHU_SZEROKOSC * 2, SZEROKOSC, PAS_RUCHU_SZEROKOSC * 4))
        pygame.draw.rect(ekran, KOLOR_DROGI, (SRODEK_X - PAS_RUCHU_SZEROKOSC * 2, 0, PAS_RUCHU_SZEROKOSC * 4, WYSOKOSC))
        
def main():
    pygame.init()
    ekran = pygame.display.set_mode((SZEROKOSC, WYSOKOSC))
    pygame.display.set_caption("Symulacja skrzyżowania (Model Agentowy)")
    zegar = pygame.time.Clock()

    skrzyzowanie = KontrolerSkrzyzowania()
    pojazdy = [] 

    uruchomiony = True
    while uruchomiony:
        dt = zegar.tick(60) / 1000.0 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                uruchomiony = False

        skrzyzowanie.aktualizuj(dt)
        
        for pojazd in pojazdy:
            pojazd.aktualizuj(skrzyzowanie, pojazdy)
            
        if random.randint(0, 100) < 1:
              pojazdy.append(Pojazd(0, SRODEK_Y - PAS_RUCHU_SZEROKOSC, 1, 0))
        
        if random.randint(0, 100) < 1:
              pojazdy.append(Pojazd(SRODEK_X - PAS_RUCHU_SZEROKOSC, 0, 0, 1))
              
        pojazdy = [p for p in pojazdy if p.x < SZEROKOSC and p.y < WYSOKOSC and p.x > -40 and p.y > -40]

        ekran.fill(KOLOR_TLA)
        skrzyzowanie.narysuj_drogi(ekran)
        skrzyzowanie.rysuj(ekran)
        
        for pojazd in pojazdy:
            pojazd.rysuj(ekran)
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()