from game import PylosGame


def parse_coords(text):
    parts = text.split(',')
    if len(parts) != 3:
        raise ValueError
    return tuple(int(p) for p in parts)


def main():
    game = PylosGame()
    while True:
        print(game)
        if game.top_filled():
            winner = "White" if game.board[-1][0, 0] == 1 else "Black"
            print(f"{winner} wins by completing the pyramid!")
            break
        if not game.has_move():
            winner = "White" if game.turn == -1 else "Black"
            print(f"{winner} wins. Opponent has no moves left.")
            break
        player = "White" if game.turn == 1 else "Black"
        print(f"{player}'s turn. Reserve pieces: {game.reserves[game.turn]}")
        cmd = input("Enter command (place l,r,c | raise sl,sr,sc dl,dr,dc): ").strip()
        if not cmd:
            continue
        if cmd.lower() == 'quit':
            break
        tokens = cmd.replace('->', ' ').replace('|', ' ').split()
        try:
            if tokens[0] == 'place' and len(tokens) == 2:
                lvl, r, c = parse_coords(tokens[1])
                if not game.place(lvl, r, c):
                    print("Invalid placement")
                    continue
            elif tokens[0] == 'raise' and len(tokens) == 3:
                sl, sr, sc = parse_coords(tokens[1])
                dl, dr, dc = parse_coords(tokens[2])
                if not game.raise_piece(sl, sr, sc, dl, dr, dc):
                    print("Invalid raise move")
                    continue
            else:
                print("Unrecognized command")
                continue
        except Exception:
            print("Invalid input format")
            continue
        # removal phase
        if game.check_for_removal():
            for i in range(2):
                rem = input(f"You may remove piece {i+1} (level,row,col) or press Enter to skip: ").strip()
                if not rem:
                    break
                try:
                    lvl, r, c = parse_coords(rem)
                    if not game.remove(lvl, r, c):
                        print("Cannot remove that piece")
                        continue
                except Exception:
                    print("Invalid coordinates")
                    continue
        # switch turn
        game.turn *= -1


if __name__ == '__main__':
    main()
